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
    [Tooltip("Fichier fusionné anatomy_label_weighted (*_lw.vrdf). Si renseigné, on ignore les deux champs ci-dessous.")]
    public string vrdfFusedFileName = "scene_lw.vrdf";

    [Tooltip("Ancien mode: fichier labelmap (anatomy_label / labelmap TF). Ignoré si vrdfFusedFileName est renseigné.")]
    public string vrdfLabelsFileName = "scene_labels.vrdf";

    [Tooltip("Ancien mode: fichier weight (activity_weight / continuous TF). Peut être vide. Ignoré si vrdfFusedFileName est renseigné.")]
    public string vrdfWeightsFileName = "scene_weights.vrdfw";

    [Header("Raymarch material (must match shader)")]
    public Material volumeMaterial;

    [Header("Debug / Quality")]
    [Tooltip("false = rendu clinique doux (soft LUT)\ntrue = rendu segmentation dure (hard LUT)")]
    public bool useHardTF = false;

    public bool verboseDebug = false;

    // runtime GPU data
    private Texture3D volumeTexLabels;   // label volume (ou labelTexture)
    private Texture3D volumeTexWeights;  // weight volume (ou weightTexture)

    private Texture2D tfTexCurrent;
    private Texture2D _labelCtrlTex;
    private Color[]   _labelCtrlPixels;

    [NonSerialized] public List<VolumeLabelInfoRuntime> labelInfos = new List<VolumeLabelInfoRuntime>();

    private VRDFVolumeData _labelsData;   // dans le cas fusionné: ceci contient label+weight
    private VRDFVolumeData _weightsData;  // ancien mode uniquement

    private static Texture3D _blackTex3D;
    private static Texture3D BlackTex3D
    {
        get
        {
            if (_blackTex3D == null)
            {
                Color black = new Color(0, 0, 0, 0);
                _blackTex3D = new Texture3D(1, 1, 1, TextureFormat.RFloat, false);
                _blackTex3D.SetPixel(0, 0, 0, black);
                _blackTex3D.Apply(false, false);
            }
            return _blackTex3D;
        }
    }

    void Start()
    {
        if (!string.IsNullOrEmpty(vrdfFusedFileName))
        {
            // ===== Nouveau mode : un seul fichier lw.vrdf =====
            string fusedPath = Path.Combine(Application.streamingAssetsPath, vrdfFusedFileName);
            _labelsData = VRDFLoader.LoadFromFile(fusedPath);
            VRDFLoader.BuildUnityTextures(_labelsData);

            // Dans ce cas, _labelsData contient labelTexture + weightTexture
            volumeTexLabels  = _labelsData.labelTexture;
            volumeTexWeights = _labelsData.weightTexture;
            _weightsData     = null; // pas besoin, tout est dans _labelsData

            // LUT
            tfTexCurrent = useHardTF ? _labelsData.tfLUTTextureHard
                                     : _labelsData.tfLUTTextureSoft;
        }
        else
        {
            // ===== Ancien mode : deux fichiers séparés =====
            string pathLabels = Path.Combine(Application.streamingAssetsPath, vrdfLabelsFileName);
            _labelsData = VRDFLoader.LoadFromFile(pathLabels);
            VRDFLoader.BuildUnityTextures(_labelsData);

            volumeTexLabels = _labelsData.volumeTexture;

            _weightsData = null;
            volumeTexWeights = null;
            if (!string.IsNullOrEmpty(vrdfWeightsFileName))
            {
                string pathWeights = Path.Combine(Application.streamingAssetsPath, vrdfWeightsFileName);
                if (File.Exists(pathWeights))
                {
                    _weightsData = VRDFLoader.LoadFromFile(pathWeights);
                    VRDFLoader.BuildUnityTextures(_weightsData);
                    volumeTexWeights = _weightsData.volumeTexture;
                }
                else if (verboseDebug)
                {
                    Debug.LogWarning("[VolumeDVR] No weights file found at " + pathWeights + " (will render without modulation)");
                }
            }

            tfTexCurrent = useHardTF ? _labelsData.tfLUTTextureHard
                                     : _labelsData.tfLUTTextureSoft;
        }

        ApplyToMaterial();
        InitLabelCtrlTexture();
        BuildLabelInfosFromData();
        FitVolumeScaleFromSpacing(_labelsData);

        if (verboseDebug)
        {
            Debug.Log($"[VolumeDVR] Loaded main vol {DimToString(_labelsData.meta.dim)} mode={_labelsData.meta.mode} tf={_labelsData.tf.type}");

            if (_weightsData != null)
                Debug.Log($"[VolumeDVR] Loaded WEIGHT vol {DimToString(_weightsData.meta.dim)} mode={_weightsData.meta.mode} tf={_weightsData.tf.type}");

            Debug.Log($"[VolumeDVR] worldPos={transform.position} worldScale={transform.lossyScale}");

            string tfDescSoft = _labelsData.tfLUTTextureSoft ? (_labelsData.tfLUTTextureSoft.width + "x" + _labelsData.tfLUTTextureSoft.height + " soft") : "null";
            string tfDescHard = _labelsData.tfLUTTextureHard ? (_labelsData.tfLUTTextureHard.width + "x" + _labelsData.tfLUTTextureHard.height + " hard") : "null";
            Debug.Log($"[VolumeDVR] TF soft={tfDescSoft} hard={tfDescHard}, usingHard={useHardTF}");
        }
    }

    private string DimToString(int[] d)
    {
        if (d == null || d.Length < 3) return "???";
        return d[0] + "x" + d[1] + "x" + d[2];
    }

    public void SetTFMode(bool hard)
    {
        useHardTF = hard;
        tfTexCurrent = useHardTF ? _labelsData.tfLUTTextureHard
                                 : _labelsData.tfLUTTextureSoft;

        if (volumeMaterial != null && tfTexCurrent != null)
        {
            volumeMaterial.SetTexture("_TFTex", tfTexCurrent);
        }

        BuildLabelInfosFromData();
    }

    private void ApplyToMaterial()
    {
        if (volumeMaterial == null)
        {
            Debug.LogError("[VolumeDVR] volumeMaterial not set.");
            return;
        }
        if (_labelsData == null || _labelsData.meta == null)
        {
            Debug.LogError("[VolumeDVR] Missing labels data.");
            return;
        }

        var d = _labelsData;

        int dimX = d.meta.dim[0];
        int dimY = d.meta.dim[1];
        int dimZ = d.meta.dim[2];

        // P1/P99 : pour continuous, mais dans notre pipeline labelmap_weighted4d
        // on s'en fiche, donc ça restera (0,1)
        float p1  = 0f;
        float p99 = 1f;
        if (d.tf.type == "continuous"
            && d.meta.intensity_range != null
            && d.meta.intensity_range.Length == 2)
        {
            p1  = d.meta.intensity_range[0];
            p99 = d.meta.intensity_range[1];
        }

        // matrices
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

        // Sélection de quelles textures envoyer au shader :
        Texture3D texLbl = volumeTexLabels != null ? volumeTexLabels : (d.labelTexture != null ? d.labelTexture : BlackTex3D);
        Texture3D texW   = volumeTexWeights != null ? volumeTexWeights : (d.weightTexture != null ? d.weightTexture : null);

        volumeMaterial.SetTexture("_VolumeTexLabels", texLbl ? texLbl : BlackTex3D);

        if (texW != null)
        {
            volumeMaterial.SetTexture("_VolumeTexWeights", texW);
            volumeMaterial.SetInt("_HasWeights", 1);
        }
        else
        {
            volumeMaterial.SetTexture("_VolumeTexWeights", BlackTex3D);
            volumeMaterial.SetInt("_HasWeights", 0);
        }

        volumeMaterial.SetTexture("_TFTex", tfTexCurrent != null ? tfTexCurrent : Texture2D.blackTexture);

        // c'est un labelmap si la TF est de type "labelmap"
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
        _labelCtrlTex.wrapMode   = TextureWrapMode.Clamp;
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
        if (_labelsData == null || _labelsData.tf == null)
            return;

        Color[] lutPixels = null;
        int lutLen = 0;
        if (tfTexCurrent != null)
        {
            lutPixels = tfTexCurrent.GetPixels();
            lutLen = lutPixels.Length;
        }

        if (_labelsData.tf.entries != null && _labelsData.tf.entries.Count > 0)
        {
            foreach (var e in _labelsData.tf.entries)
            {
                int lbl = Mathf.RoundToInt(e.label);

                string niceName = !string.IsNullOrEmpty(e.name) ? e.name : ("Label " + lbl);

                Color baseCol = Color.clear;
                bool gotFromLut = false;

                if (lutPixels != null && lbl >= 0 && lbl < lutLen)
                {
                    baseCol = lutPixels[lbl];
                    gotFromLut = true;
                }

                if (!gotFromLut)
                {
                    float r = (e.color != null && e.color.Length > 0) ? e.color[0] : 1f;
                    float g = (e.color != null && e.color.Length > 1) ? e.color[1] : 1f;
                    float b = (e.color != null && e.color.Length > 2) ? e.color[2] : 1f;
                    float a = e.alpha;
                    baseCol = new Color(r,g,b,a);
                }

                bool visibleDefault = baseCol.a > 0.001f;

                var info = new VolumeLabelInfoRuntime {
                    labelIndex     = lbl,
                    displayName    = niceName,
                    color          = baseCol,
                    defaultVisible = visibleDefault
                };
                labelInfos.Add(info);
            }
        }
        else if (lutPixels != null)
        {
            for (int lbl = 0; lbl < lutLen && lbl < 256; lbl++)
            {
                var c = lutPixels[lbl];
                bool visibleDefault = c.a > 0.001f;

                labelInfos.Add(new VolumeLabelInfoRuntime {
                    labelIndex     = lbl,
                    displayName    = "Label " + lbl,
                    color          = c,
                    defaultVisible = visibleDefault
                });
            }
        }

        if (verboseDebug)
        {
            Debug.Log($"[VolumeDVR] Built labelInfos ({labelInfos.Count}) using {(useHardTF ? "HARD":"SOFT")} LUT");
            foreach (var li in labelInfos)
            {
                Debug.Log($"  idx={li.labelIndex} name={li.displayName} col={li.color} visDefault={li.defaultVisible}");
            }
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
        transform.localRotation = Quaternion.Euler(-90, 0, 0);
    }

    //==================== RUNTIME INTERACTION (identique sauf qu'on pousse _LabelCtrlTex déjà fait) ====================

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
