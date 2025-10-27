using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;

[Serializable]
public class VolumeLabelInfoRuntime
{
    public int labelIndex;        // 0..255
    public string displayName;    // "Tumeur", etc.
    public Color color;           // couleur TF initiale
    public bool defaultVisible;   // alpha > 0
}

[DisallowMultipleComponent]
public class VolumeDVR : MonoBehaviour
{
    [Header("Input (.vrdf in StreamingAssets)")]
    public string vrdfFileName = "scene.vrdf";

    [Header("Raymarch material (must already be on the MeshRenderer of this object)")]
    public Material volumeMaterial;

    [Header("Debug")]
    public bool verboseDebug = false;

    // runtime GPU data
    private Texture3D volumeTex;
    private Texture2D tfTex;
    private Texture2D _labelCtrlTex;
    private Color[] _labelCtrlPixels;

    [NonSerialized] public List<VolumeLabelInfoRuntime> labelInfos = new List<VolumeLabelInfoRuntime>();

    private VRDFVolumeData _data;

    void Start()
    {
        string vrdfPath = Path.Combine(Application.streamingAssetsPath, vrdfFileName);
        _data = VRDFLoader.LoadFromFile(vrdfPath);

        VRDFLoader.BuildUnityTextures(_data);
        volumeTex = _data.volumeTexture;
        tfTex     = _data.tfLUTTexture;

        ApplyToMaterial(_data);

        InitLabelCtrlTexture();

        BuildLabelInfosFromData();

        FitVolumeScaleFromSpacing(_data);

        if (verboseDebug)
        {
            Debug.Log($"[VolumeDVR] Loaded {_data.meta.dim[0]}x{_data.meta.dim[1]}x{_data.meta.dim[2]} mode={_data.meta.mode} tf={_data.tf.type}");
            Debug.Log($"[VolumeDVR] worldPos={transform.position} worldScale={transform.lossyScale}");
            string tfDesc = (tfTex != null) ? $"{tfTex.width}x{tfTex.height}" : "null";
            Debug.Log($"[VolumeDVR] labelInfos={labelInfos.Count} entries, tfTex={tfDesc}");
        }
    }

    private void ApplyToMaterial(VRDFVolumeData d)
    {
        if (volumeMaterial == null)
        {
            Debug.LogError("[VolumeDVR] volumeMaterial not set.");
            return;
        }

        int dimX = d.meta.dim[0];
        int dimY = d.meta.dim[1];
        int dimZ = d.meta.dim[2];

        float p1  = 0f;
        float p99 = 1f;
        if (d.tf.type == "continuous" && d.meta.intensity_range != null && d.meta.intensity_range.Length == 2)
        {
            p1  = d.meta.intensity_range[0];
            p99 = d.meta.intensity_range[1];
        }

        Matrix4x4 affine = Matrix4x4.identity;
        Matrix4x4 invAffine = Matrix4x4.identity;
        if (d.meta.affine != null)
        {
            affine.SetRow(0, new Vector4(d.meta.affine[0,0], d.meta.affine[0,1], d.meta.affine[0,2], d.meta.affine[0,3]));
            affine.SetRow(1, new Vector4(d.meta.affine[1,0], d.meta.affine[1,1], d.meta.affine[1,2], d.meta.affine[1,3]));
            affine.SetRow(2, new Vector4(d.meta.affine[2,0], d.meta.affine[2,1], d.meta.affine[2,2], d.meta.affine[2,3]));
            affine.SetRow(3, new Vector4(d.meta.affine[3,0], d.meta.affine[3,1], d.meta.affine[3,2], d.meta.affine[3,3]));
            invAffine = affine.inverse;
        }

        volumeMaterial.SetTexture("_VolumeTex", volumeTex);
        volumeMaterial.SetTexture("_TFTex", tfTex != null ? tfTex : Texture2D.blackTexture);

        volumeMaterial.SetInt("_IsLabelMap", (d.tf.type == "labelmap") ? 1 : 0);
        volumeMaterial.SetFloat("_P1",  p1);
        volumeMaterial.SetFloat("_P99", p99);
        volumeMaterial.SetVector("_Dim", new Vector4(dimX, dimY, dimZ, 1f));
        volumeMaterial.SetMatrix("_Affine", affine);
        volumeMaterial.SetMatrix("_InvAffine", invAffine);

        var mr = GetComponent<MeshRenderer>();
        if (mr && mr.sharedMaterial != volumeMaterial)
            mr.sharedMaterial = volumeMaterial;
    }

    private void InitLabelCtrlTexture()
    {
        _labelCtrlTex = new Texture2D(256, 1, TextureFormat.RGBAFloat, false);
        _labelCtrlTex.wrapMode = TextureWrapMode.Clamp;
        _labelCtrlTex.filterMode = FilterMode.Point;

        _labelCtrlPixels = new Color[256];
        for (int i = 0; i < 256; i++)
            _labelCtrlPixels[i] = new Color(1f,1f,1f,1f);

        _labelCtrlTex.SetPixels(_labelCtrlPixels);
        _labelCtrlTex.Apply(false);

        volumeMaterial.SetTexture("_LabelCtrlTex", _labelCtrlTex);
    }

    private void BuildLabelInfosFromData()
    {
        labelInfos.Clear();
        if (_data == null || _data.tf == null || tfTex == null)
            return;

        Color[] lutPixels = tfTex.GetPixels();
        int lutLen = lutPixels.Length;

        Dictionary<int,string> names = new Dictionary<int,string>();
        if (_data.tf.entries != null)
        {
            foreach (var e in _data.tf.entries)
            {
                int lbl = Mathf.RoundToInt(e.label);
                if (!names.ContainsKey(lbl))
                {
                    names[lbl] = string.IsNullOrEmpty(e.name) ? ("Label " + lbl) : e.name;
                }
            }
        }

        for (int label = 0; label < lutLen && label < 256; label++)
        {
            Color baseCol = lutPixels[label];
            bool visibleDefault = baseCol.a > 0.001f;

            string niceName;
            if (!names.TryGetValue(label, out niceName))
                niceName = "Label " + label;

            var info = new VolumeLabelInfoRuntime {
                labelIndex     = label,
                displayName    = niceName,
                color          = baseCol,
                defaultVisible = visibleDefault
            };
            labelInfos.Add(info);
        }
    }

    private void FitVolumeScaleFromSpacing(VRDFVolumeData data)
    {
        var meta = data.meta;
        int dimX = meta.dim[0];
        int dimY = meta.dim[1];
        int dimZ = meta.dim[2];

        float sx = (meta.spacing_mm != null && meta.spacing_mm.Length > 0) ? meta.spacing_mm[0] : 1f;
        float sy = (meta.spacing_mm != null && meta.spacing_mm.Length > 1) ? meta.spacing_mm[1] : 1f;
        float sz = (meta.spacing_mm != null && meta.spacing_mm.Length > 2) ? meta.spacing_mm[2] : 1f;

        Vector3 sizeMeters = new Vector3(
            dimX * sx * 0.001f,
            dimY * sy * 0.001f,
            dimZ * sz * 0.001f
        );


        transform.localScale = sizeMeters;
        transform.rotation = Quaternion.Euler(-90, 0, 0);
    }


    public void SetLabelVisible(int labelIndex, bool visible)
    {
        if (!EnsureCtrlReady()) return;
        if (labelIndex < 0 || labelIndex > 255) return;
        var c = _labelCtrlPixels[labelIndex];
        c.a = visible ? 1f : 0f;
        _labelCtrlPixels[labelIndex] = c;
        _labelCtrlTex.SetPixels(_labelCtrlPixels);
        _labelCtrlTex.Apply(false);
    }

    public void SetLabelOpacity(int labelIndex, float opacity01)
    {
        if (!EnsureCtrlReady()) return;
        if (labelIndex < 0 || labelIndex > 255) return;
        var c = _labelCtrlPixels[labelIndex];
        c.a = Mathf.Clamp01(opacity01);
        _labelCtrlPixels[labelIndex] = c;
        _labelCtrlTex.SetPixels(_labelCtrlPixels);
        _labelCtrlTex.Apply(false);
    }

    public void SetLabelTint(int labelIndex, Color tintRGB)
    {
        if (!EnsureCtrlReady()) return;
        if (labelIndex < 0 || labelIndex > 255) return;
        var c = _labelCtrlPixels[labelIndex];
        c.r = tintRGB.r;
        c.g = tintRGB.g;
        c.b = tintRGB.b;
        _labelCtrlPixels[labelIndex] = c;
        _labelCtrlTex.SetPixels(_labelCtrlPixels);
        _labelCtrlTex.Apply(false);
    }

    public void SoloLabel(int soloIndex)
    {
        if (!EnsureCtrlReady()) return;
        for (int i = 0; i < 256; i++)
        {
            var c = _labelCtrlPixels[i];
            c.a = (i == soloIndex) ? 1f : 0f;
            _labelCtrlPixels[i] = c;
        }
        _labelCtrlTex.SetPixels(_labelCtrlPixels);
        _labelCtrlTex.Apply(false);
    }

    public void ShowAll()
    {
        if (!EnsureCtrlReady()) return;
        for (int i = 0; i < 256; i++)
            _labelCtrlPixels[i] = new Color(1f,1f,1f,1f);

        _labelCtrlTex.SetPixels(_labelCtrlPixels);
        _labelCtrlTex.Apply(false);
    }

    private bool EnsureCtrlReady()
    {
        if (_labelCtrlTex == null || _labelCtrlPixels == null)
        {
            Debug.LogWarning("[VolumeDVR] _labelCtrlTex not ready.");
            return false;
        }
        return true;
    }
}
