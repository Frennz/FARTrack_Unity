using UnityEngine;
using UnityEditor;
using System.Linq;

namespace FarTrack
{
    [CustomEditor(typeof(FARTrackSentis))]
    public class FARTrackSentisEditor : Editor
    {
        private string[] availableCameras = new string[0];
        private int selectedCameraIndex = 0;

        private void OnEnable()
        {
            if (WebCamTexture.devices != null)
            {
                availableCameras = WebCamTexture.devices.Select(d => d.name).ToArray();
            }
            
            FARTrackSentis tracker = (FARTrackSentis)target;
            selectedCameraIndex = Mathf.Max(0, System.Array.IndexOf(availableCameras, tracker.deviceName));
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            FARTrackSentis tracker = (FARTrackSentis)target;

            DrawPropertiesExcluding(serializedObject, "m_Script", "deviceName");

            GUILayout.Space(10);
            EditorGUILayout.LabelField("Camera Selection", EditorStyles.boldLabel);

            if (availableCameras.Length > 0)
            {
                selectedCameraIndex = EditorGUILayout.Popup("Select Webcam", selectedCameraIndex, availableCameras);
                
                SerializedProperty deviceNameProp = serializedObject.FindProperty("deviceName");
                if (deviceNameProp != null)
                {
                    deviceNameProp.stringValue = availableCameras[selectedCameraIndex];
                }
            }
            else
            {
                EditorGUILayout.HelpBox("No webcams found! Make sure a camera is plugged in.", MessageType.Warning);
                
                SerializedProperty deviceNameProp = serializedObject.FindProperty("deviceName");
                EditorGUILayout.PropertyField(deviceNameProp, new GUIContent("Manual Device Name"));
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}
