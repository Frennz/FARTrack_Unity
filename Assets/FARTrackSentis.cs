using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using Unity.InferenceEngine;

namespace FarTrack
{
    // ==========================================
    // STRUCTS & UTILS
    // ==========================================
    public struct PixelBox
    {
        public float X, Y, W, H;
        public PixelBox(float x, float y, float w, float h) { X = x; Y = y; W = w; H = h; }
        public float CenterX => X + W * 0.5f;
        public float CenterY => Y + H * 0.5f;
    }

    public struct CropInfo
    {
        public float CropX0, CropY0, CropSize;
        public float ResizeFactor;
    }

    public static class CropUtils
    {
        public static CropInfo ComputeCrop(PixelBox box, float areaFactor, int outputSize)
        {
            float cropSize = Mathf.Sqrt(box.W * box.H) * areaFactor;
            if (cropSize < 1f) cropSize = 1f;

            float x0 = box.CenterX - cropSize * 0.5f;
            float y0 = box.CenterY - cropSize * 0.5f;

            return new CropInfo
            {
                CropX0 = x0,
                CropY0 = y0,
                CropSize = cropSize,
                ResizeFactor = outputSize / cropSize
            };
        }

        public static void BlitCrop(Texture source, int sourceWidth, int sourceHeight, CropInfo crop, RenderTexture dest)
        {
            float u0 = crop.CropX0 / sourceWidth;
            float v0 = 1f - (crop.CropY0 + crop.CropSize) / sourceHeight;
            float uSize = crop.CropSize / sourceWidth;
            float vSize = crop.CropSize / sourceHeight;

            Vector2 scale = new Vector2(uSize, vSize);
            Vector2 offset = new Vector2(u0, v0);
            Graphics.Blit(source, dest, scale, offset);
        }

        public static PixelBox MapPredictionToImage(float x1_bin, float y1_bin, float x2_bin, float y2_bin, CropInfo crop, int bins = 600)
        {
            float x1 = (x1_bin / (bins - 1)) - 0.5f;
            float y1 = (y1_bin / (bins - 1)) - 0.5f;
            float x2 = (x2_bin / (bins - 1)) - 0.5f;
            float y2 = (y2_bin / (bins - 1)) - 0.5f;

            float w_norm = x2 - x1;
            float h_norm = y2 - y1;
            float cx_norm = x1 + (w_norm * 0.5f);
            float cy_norm = y1 + (h_norm * 0.5f);

            float cx_local = cx_norm + 0.5f;
            float cy_local = cy_norm + 0.5f;

            float cx = crop.CropX0 + cx_local * crop.CropSize;
            float cy = crop.CropY0 + cy_local * crop.CropSize;
            float w = w_norm * crop.CropSize;
            float h = h_norm * crop.CropSize;

            return new PixelBox(cx - w * 0.5f, cy - h * 0.5f, w, h);
        }
    }

    // ==========================================
    // MONOBEHAVIOUR (Unified UI & Tracker)
    // ==========================================
    public class FARTrackSentis : MonoBehaviour, IPointerDownHandler, IDragHandler, IPointerUpHandler
    {
        [Header("Webcam request")]
        public int requestedWidth = 1920;
        public int requestedHeight = 1080;
        public int requestedFPS = 30;
        public string deviceName = ""; // Empty = first available

        [Header("Scene Refs")]
        public RawImage videoImage;
        public AspectRatioFitter aspectFitter;
        public ModelAsset trackerModel;

        [Header("Tracker Settings")]
        [Tooltip("Coordinate bins output by the ONNX model. Tiny=600, Base=4000.")]
        public int coordinateBins = 600;
        [Tooltip("How much to smooth the box. 0 = instant, 0.9 = slow/smooth")]
        [Range(0f, 0.95f)]
        public float smoothingFactor = 0.5f;
        [Tooltip("If the box shrinks below this pixel area, tracking is dropped.")]
        public float minTrackingArea = 400f;

        [Header("UI Customization")]
        public Color selectionColor = Color.green;
        public Color trackingColor = Color.red;
        public float lineThickness = 3f;

        [Header("Debug")]
        public RawImage debugSearchCropView;

        // UI state
        private RectTransform selectionBoxRT;
        private RectTransform trackingBoxRT;
        private WebCamTexture _webcam;
        private bool _mirrored;
        private int _texW, _texH;
        private bool _isTracking;
        private Vector2 _dragStartLocal;

        // Tracker state
        private const int TemplateSize = 112;
        private const int SearchSize = 224;
        private const int NumTemplates = 5;
        private readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        public float TemplateFactor = 2.0f;
        public float SearchFactor = 4.0f;
        public int TemplateUpdateInterval = 30;

        private Model _runtimeModel;
        private Worker _worker;
        private RenderTexture _templateCropRT;
        private RenderTexture _searchCropRT;
        private Texture2D _readbackTex;
        private float[][] _templateBuffers;
        private int _nextTemplateSlot;
        private int _framesSinceInit;
        private PixelBox CurrentBox;
        private bool IsTrackerInitialized;

        void Start()
        {
            // UI Initialization
            selectionBoxRT = CreateBorderRect("SelectionBox", selectionColor, lineThickness);
            trackingBoxRT = CreateBorderRect("TrackingBox", trackingColor, lineThickness);
            
            selectionBoxRT.gameObject.SetActive(false);
            trackingBoxRT.gameObject.SetActive(false);

            string chosenDevice = string.IsNullOrEmpty(deviceName) && WebCamTexture.devices.Length > 0
                ? WebCamTexture.devices[0].name
                : deviceName;

            _webcam = new WebCamTexture(chosenDevice, requestedWidth, requestedHeight, requestedFPS);
            _webcam.wrapMode = TextureWrapMode.Clamp;
            _webcam.Play();

            // Tracker Initialization
            _runtimeModel = ModelLoader.Load(trackerModel);
            
            // Force dynamic shapes to prevent Unity InferenceEngine static optimizer freeze
            if (_runtimeModel.inputs.Count > 0)
            {
                var inp0 = _runtimeModel.inputs[0];
                inp0.shape = new DynamicTensorShape(-1, 3, -1, -1);
                _runtimeModel.inputs[0] = inp0;
            }
            if (_runtimeModel.inputs.Count > 1)
            {
                var inp1 = _runtimeModel.inputs[1];
                inp1.shape = new DynamicTensorShape(-1, 3, -1, -1);
                _runtimeModel.inputs[1] = inp1;
            }

            _worker = new Worker(_runtimeModel, BackendType.GPUCompute);
            _templateCropRT = new RenderTexture(TemplateSize, TemplateSize, 0, RenderTextureFormat.ARGB32);
            _templateCropRT.Create();
            _searchCropRT = new RenderTexture(SearchSize, SearchSize, 0, RenderTextureFormat.ARGB32);
            _searchCropRT.Create();
            _readbackTex = new Texture2D(SearchSize, SearchSize, TextureFormat.RGBA32, false);

            _templateBuffers = new float[NumTemplates][];
            for (int i = 0; i < NumTemplates; i++)
                _templateBuffers[i] = new float[3 * TemplateSize * TemplateSize];

            WarmupTracker();
        }

        void WarmupTracker()
        {
            Debug.Log("[FARTrack] Warming up GPU Compute Shaders... Unity WILL FREEZE for 30-60 seconds! Please wait...");
            float[] dummyTemplate = new float[NumTemplates * 3 * TemplateSize * TemplateSize];
            float[] dummySearch = new float[1 * 3 * SearchSize * SearchSize];

            using var templatesTensor = new Tensor<float>(new TensorShape(NumTemplates, 3, TemplateSize, TemplateSize), dummyTemplate);
            using var searchTensor = new Tensor<float>(new TensorShape(1, 3, SearchSize, SearchSize), dummySearch);

            _worker.SetInput("templates", templatesTensor);
            _worker.SetInput("search", searchTensor);
            _worker.Schedule();

            var outputTensor = _worker.PeekOutput("pred_boxes") as Tensor<float>;
            float[] pred = outputTensor.DownloadToArray();
            Debug.Log("[FARTrack] Warmup complete! Shaders compiled.");
        }

        void Update()
        {
            if (_webcam == null || !_webcam.didUpdateThisFrame || _webcam.width < 100) return;

            if (videoImage.texture == null)
            {
                videoImage.texture = _webcam;
            }

            if (_texW != _webcam.width || _texH != _webcam.height)
            {
                _texW = _webcam.width;
                _texH = _webcam.height;
                _mirrored = _webcam.videoVerticallyMirrored;
                aspectFitter.aspectRatio = _texW / (float)_texH;
                videoImage.rectTransform.localEulerAngles = new Vector3(0, 0, -_webcam.videoRotationAngle);
            }

            if (_isTracking && IsTrackerInitialized)
            {
                var box = RunTrackingLoop(_webcam, _texW, _texH);
                
                if (float.IsNaN(box.X) || float.IsInfinity(box.X) || float.IsNaN(box.W))
                {
                    Debug.LogError("[FARTrack] NaN returned. Resetting tracking.");
                    ResetTracking();
                    return;
                }

                DrawOverlay(trackingBoxRT, box);

                if (debugSearchCropView != null)
                    debugSearchCropView.texture = _searchCropRT;
            }
        }

        // ==========================================
        // SENTIS TRACKING LOGIC
        // ==========================================
        void InitializeTracker(Texture sourceTex, int sourceW, int sourceH, PixelBox box)
        {
            var crop = CropUtils.ComputeCrop(box, TemplateFactor, TemplateSize);
            CropUtils.BlitCrop(sourceTex, sourceW, sourceH, crop, _templateCropRT);
            float[] templateData = ReadNormalizedCHW(_templateCropRT, TemplateSize);

            for (int i = 0; i < NumTemplates; i++)
                _templateBuffers[i] = (float[])templateData.Clone();

            _nextTemplateSlot = 0;
            _framesSinceInit = 0;
            CurrentBox = box;
            IsTrackerInitialized = true;
        }

        PixelBox RunTrackingLoop(Texture sourceTex, int sourceW, int sourceH)
        {
            var crop = CropUtils.ComputeCrop(CurrentBox, SearchFactor, SearchSize);
            CropUtils.BlitCrop(sourceTex, sourceW, sourceH, crop, _searchCropRT);
            float[] searchData = ReadNormalizedCHW(_searchCropRT, SearchSize);

            float[] templatesFlat = new float[NumTemplates * 3 * TemplateSize * TemplateSize];
            int perTemplate = 3 * TemplateSize * TemplateSize;
            for (int i = 0; i < NumTemplates; i++)
                System.Array.Copy(_templateBuffers[i], 0, templatesFlat, i * perTemplate, perTemplate);

            using var templatesTensor = new Tensor<float>(new TensorShape(NumTemplates, 3, TemplateSize, TemplateSize), templatesFlat);
            using var searchTensor = new Tensor<float>(new TensorShape(1, 3, SearchSize, SearchSize), searchData);

            _worker.SetInput("templates", templatesTensor);
            _worker.SetInput("search", searchTensor);
            _worker.Schedule();

            var outputTensor = _worker.PeekOutput("pred_boxes") as Tensor<float>;
            float[] pred = outputTensor.DownloadToArray();

            // Decode coordinates using the configurable bins value
            var newBox = CropUtils.MapPredictionToImage(pred[0], pred[1], pred[2], pred[3], crop, coordinateBins);
            var clampedNewBox = ClampToImage(newBox, sourceW, sourceH);

            // Apply Exponential Moving Average (EMA) smoothing
            if (_framesSinceInit > 0)
            {
                CurrentBox.X = Mathf.Lerp(clampedNewBox.X, CurrentBox.X, smoothingFactor);
                CurrentBox.Y = Mathf.Lerp(clampedNewBox.Y, CurrentBox.Y, smoothingFactor);
                CurrentBox.W = Mathf.Lerp(clampedNewBox.W, CurrentBox.W, smoothingFactor);
                CurrentBox.H = Mathf.Lerp(clampedNewBox.H, CurrentBox.H, smoothingFactor);
            }
            else
            {
                CurrentBox = clampedNewBox;
            }

            // Tracking Loss Heuristic: If box shrinks too small, drop tracking
            if (CurrentBox.W * CurrentBox.H < minTrackingArea)
            {
                Debug.Log("[FARTrack] Tracking lost (confidence/size heuristic triggered).");
                ResetTracking();
                return CurrentBox;
            }

            _framesSinceInit++;
            MaybeUpdateTemplate(sourceTex, sourceW, sourceH);

            return CurrentBox;
        }

        void MaybeUpdateTemplate(Texture sourceTex, int sourceW, int sourceH)
        {
            if (TemplateUpdateInterval <= 0) return;
            if (_framesSinceInit % TemplateUpdateInterval != 0) return;

            var crop = CropUtils.ComputeCrop(CurrentBox, TemplateFactor, TemplateSize);
            CropUtils.BlitCrop(sourceTex, sourceW, sourceH, crop, _templateCropRT);
            
            // NOTE: We NEVER overwrite template 0! Template 0 is the original ground truth 
            // from the drag selection. Overwriting it will cause the tracker to drift and fail.
            if (_nextTemplateSlot == 0) _nextTemplateSlot = 1;
            
            _templateBuffers[_nextTemplateSlot] = ReadNormalizedCHW(_templateCropRT, TemplateSize);
            _nextTemplateSlot++;
            if (_nextTemplateSlot >= NumTemplates) _nextTemplateSlot = 1;
        }

        static PixelBox ClampToImage(PixelBox box, int w, int h)
        {
            float x = Mathf.Clamp(box.X, 0, w - 2);
            float y = Mathf.Clamp(box.Y, 0, h - 2);
            float bw = Mathf.Clamp(box.W, 2, w - x);
            float bh = Mathf.Clamp(box.H, 2, h - y);
            return new PixelBox(x, y, bw, bh);
        }

        float[] ReadNormalizedCHW(RenderTexture rt, int size)
        {
            var prevActive = RenderTexture.active;
            RenderTexture.active = rt;
            if (_readbackTex.width != size || _readbackTex.height != size)
                _readbackTex.Reinitialize(size, size);
            _readbackTex.ReadPixels(new Rect(0, 0, size, size), 0, 0);
            _readbackTex.Apply(false);
            RenderTexture.active = prevActive;

            var pixels = _readbackTex.GetPixels32();
            var data = new float[3 * size * size];
            int plane = size * size;

            for (int y = 0; y < size; y++)
            {
                int destY = size - 1 - y;
                int rowOffset = y * size;
                int destRowOffset = destY * size;
                for (int x = 0; x < size; x++)
                {
                    var p = pixels[rowOffset + x];
                    int idx = destRowOffset + x;
                    data[0 * plane + idx] = (p.r / 255f - Mean[0]) / Std[0];
                    data[1 * plane + idx] = (p.g / 255f - Mean[1]) / Std[1];
                    data[2 * plane + idx] = (p.b / 255f - Mean[2]) / Std[2];
                }
            }
            return data;
        }

        // ==========================================
        // DRAG & UI LOGIC
        // ==========================================
        public void OnPointerDown(PointerEventData e)
        {
            if (_texW < 100) return;

            _isTracking = false;
            trackingBoxRT.gameObject.SetActive(false);

            RectTransformUtility.ScreenPointToLocalPointInRectangle(videoImage.rectTransform, e.position, e.pressEventCamera, out _dragStartLocal);
            selectionBoxRT.gameObject.SetActive(true);
            SetBoxFromLocalCorners(selectionBoxRT, _dragStartLocal, _dragStartLocal);
        }

        public void OnDrag(PointerEventData e)
        {
            if (_texW < 100) return;

            RectTransformUtility.ScreenPointToLocalPointInRectangle(videoImage.rectTransform, e.position, e.pressEventCamera, out var current);
            SetBoxFromLocalCorners(selectionBoxRT, _dragStartLocal, current);
        }

        public void OnPointerUp(PointerEventData e)
        {
            if (_texW < 100) return;

            RectTransformUtility.ScreenPointToLocalPointInRectangle(videoImage.rectTransform, e.position, e.pressEventCamera, out var current);
            selectionBoxRT.gameObject.SetActive(false);

            var pxA = LocalPointToPixel(_dragStartLocal);
            var pxB = LocalPointToPixel(current);

            float x = Mathf.Min(pxA.x, pxB.x);
            float y = Mathf.Min(pxA.y, pxB.y);
            float w = Mathf.Abs(pxB.x - pxA.x);
            float h = Mathf.Abs(pxB.y - pxA.y);

            if (w < 5 || h < 5) return;

            var box = new PixelBox(x, y, w, h);
            InitializeTracker(_webcam, _texW, _texH, box);
            trackingBoxRT.gameObject.SetActive(true);
            DrawOverlay(trackingBoxRT, box);
            _isTracking = true;
        }

        public void ResetTracking()
        {
            _isTracking = false;
            trackingBoxRT.gameObject.SetActive(false);
        }

        Vector2 LocalPointToPixel(Vector2 local)
        {
            var rect = videoImage.rectTransform.rect;
            float u = local.x / rect.width + 0.5f;
            float v = local.y / rect.height + 0.5f;
            if (_mirrored) u = 1f - u;

            float px = u * _texW;
            float py = (1f - v) * _texH;
            return new Vector2(px, py);
        }

        Vector2 PixelToLocalPoint(float px, float py)
        {
            var rect = videoImage.rectTransform.rect;
            float u = px / _texW;
            float v = 1f - py / _texH;
            if (_mirrored) u = 1f - u;

            float localX = (u - 0.5f) * rect.width;
            float localY = (v - 0.5f) * rect.height;
            return new Vector2(localX, localY);
        }

        void SetBoxFromLocalCorners(RectTransform box, Vector2 a, Vector2 b)
        {
            Vector2 min = Vector2.Min(a, b);
            Vector2 max = Vector2.Max(a, b);
            box.anchoredPosition = (min + max) * 0.5f;
            box.sizeDelta = max - min;
        }

        void DrawOverlay(RectTransform box, PixelBox pixelBox)
        {
            var topLeft = PixelToLocalPoint(pixelBox.X, pixelBox.Y);
            var bottomRight = PixelToLocalPoint(pixelBox.X + pixelBox.W, pixelBox.Y + pixelBox.H);
            SetBoxFromLocalCorners(box, topLeft, bottomRight);
        }

        private void CreateLine(RectTransform parent, string objName, Color color, float thickness, Vector2 anchorMin, Vector2 anchorMax, Vector2 pivot)
        {
            GameObject lineObj = new GameObject(objName);
            lineObj.transform.SetParent(parent, false);
            RectTransform rt = lineObj.AddComponent<RectTransform>();
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            rt.pivot = pivot;
            rt.anchoredPosition = Vector2.zero;
            rt.sizeDelta = (anchorMin.x == anchorMax.x) ? new Vector2(thickness, 0) : new Vector2(0, thickness);

            Image img = lineObj.AddComponent<Image>();
            img.color = color;
        }

        private RectTransform CreateBorderRect(string objName, Color color, float thickness)
        {
            GameObject rectObj = new GameObject(objName);
            rectObj.transform.SetParent(this.transform as RectTransform, false);
            RectTransform rt = rectObj.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(0.5f, 0.5f);
            rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.pivot = new Vector2(0.5f, 0.5f);

            CreateLine(rt, "Top", color, thickness, new Vector2(0, 1), new Vector2(1, 1), new Vector2(0.5f, 1));
            CreateLine(rt, "Bottom", color, thickness, new Vector2(0, 0), new Vector2(1, 0), new Vector2(0.5f, 0));
            CreateLine(rt, "Left", color, thickness, new Vector2(0, 0), new Vector2(0, 1), new Vector2(0, 0.5f));
            CreateLine(rt, "Right", color, thickness, new Vector2(1, 0), new Vector2(1, 1), new Vector2(1, 0.5f));

            return rt;
        }

        void OnDestroy()
        {
            _worker?.Dispose();
            if (_readbackTex != null) Destroy(_readbackTex);
            if (_templateCropRT != null) _templateCropRT.Release();
            if (_searchCropRT != null) _searchCropRT.Release();
            if (_webcam != null) _webcam.Stop();
        }
    }
}
