using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class VRDFMeta
{
    public int[] dim;                  // [dimX, dimY, dimZ]
    public float[] spacing_mm;         // [sx, sy, sz]
    public string dtype;               // "float32"
    public float[] intensity_range;    // [min, max]
    public float[,] affine;            // 4x4
    public string mode;                // "labelmap"|"continuous"|"anatomy_label"|"activity_weight"|"anatomy_label_weighted"
    public string endianness;
    public string order;

    public int channels;                              // ex: 2
    public List<string> channel_meaning;              // ex: ["labelmap","weight01"]
}


[Serializable]
public class VRDFTFEntry
{
    public float label;
    public float[] color;   // [r,g,b]
    public float alpha;     // a
    public string name;     // class name
}

[Serializable]
public class VRDFTransferFunction
{
    [Serializable]
    public class TFPoint
    {
        public float x;
        public float[] color;
        public float alpha;
    }

    public string type;  // "labelmap" | "continuous"
    public List<VRDFTFEntry> entries;
    public List<TFPoint> curve;
    public string origin;
    public List<string> channel_names_hint;
}

public class VRDFVolumeData
{
    public VRDFMeta meta;
    public VRDFTransferFunction tf;

    public float[] volumeData;
    public float[] labelData; 
    public float[] weightData;

    public Texture3D volumeTexture;
    public Texture3D labelTexture;
    public Texture3D weightTexture;

    public Texture2D tfLUTTextureSoft;
    public Texture2D tfLUTTextureHard;
}


static class VRDFBinaryUtil
{
    public static ulong ReadUInt64LE(BinaryReader br)
    {
        byte[] b = br.ReadBytes(8);
        if (b.Length != 8)
            throw new EndOfStreamException("Unexpected EOF while reading uint64.");
        if (!BitConverter.IsLittleEndian)
            Array.Reverse(b);
        return BitConverter.ToUInt64(b, 0);
    }
}

public static class VRDFLoader
{
    [Serializable]
    private class MetaWrapper
    {
        public int[] dim;
        public float[] spacing_mm;
        public string dtype;
        public float[] intensity_range;
        public float[][] affine;
        public string mode;
        public string endianness;
        public string order;

        public int channels;
        public List<string> channel_meaning;
    }

    [Serializable]
    private class TFWrapper
    {
        public string type;
        public List<VRDFTFEntry> entries;
        public List<VRDFTransferFunction.TFPoint> curve;
        public string origin;
        public List<string> channel_names_hint;
    }

    public static VRDFVolumeData LoadFromFile(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("VRDF file not found", path);

        using (var fs = File.OpenRead(path))
        using (var br = new BinaryReader(fs))
        {
            // ===== header commun =====
            byte[] magicBytes = br.ReadBytes(8);
            if (magicBytes.Length != 8) throw new Exception("File too short (no magic)");
            string magic = Encoding.ASCII.GetString(magicBytes);
            if (magic != "VRDF0001")
                throw new Exception($"Bad magic '{magic}', expected 'VRDF0001'.");

            ulong totalSize = VRDFBinaryUtil.ReadUInt64LE(br);

            // meta block
            ulong metaLen = VRDFBinaryUtil.ReadUInt64LE(br);
            byte[] metaBytes = br.ReadBytes((int)metaLen);
            if ((ulong)metaBytes.Length != metaLen)
                throw new Exception("EOF in meta block");
            string metaJson = Encoding.UTF8.GetString(metaBytes);

            // tf block
            ulong tfLen = VRDFBinaryUtil.ReadUInt64LE(br);
            byte[] tfBytes = br.ReadBytes((int)tfLen);
            if ((ulong)tfBytes.Length != tfLen)
                throw new Exception("EOF in TF block");
            string tfJson = Encoding.UTF8.GetString(tfBytes);

            // raw block
            ulong rawLen = VRDFBinaryUtil.ReadUInt64LE(br);
            byte[] rawBytes = br.ReadBytes((int)rawLen);
            if ((ulong)rawBytes.Length != rawLen)
                throw new Exception("EOF in RAW block");

            // ===== parse meta =====
            MetaWrapper metaTemp = JsonUtility.FromJson<MetaWrapper>(metaJson);
            if (metaTemp == null) throw new Exception("Failed to parse meta JSON");

            VRDFMeta meta = new VRDFMeta();
            meta.dim = metaTemp.dim;
            meta.spacing_mm = metaTemp.spacing_mm;
            meta.dtype = metaTemp.dtype;
            meta.intensity_range = metaTemp.intensity_range;
            meta.mode = metaTemp.mode;
            meta.endianness = metaTemp.endianness;
            meta.order = metaTemp.order;
            meta.channels = metaTemp.channels;
            meta.channel_meaning = metaTemp.channel_meaning ?? new List<string>();

            meta.affine = new float[4,4];
            if (metaTemp.affine != null && metaTemp.affine.Length == 4)
            {
                for (int r = 0; r < 4; r++)
                {
                    if (metaTemp.affine[r] != null && metaTemp.affine[r].Length == 4)
                    {
                        for (int c = 0; c < 4; c++)
                            meta.affine[r,c] = metaTemp.affine[r][c];
                    }
                }
            }

            int dimX = meta.dim[0];
            int dimY = meta.dim[1];
            int dimZ = meta.dim[2];
            int voxelCount = dimX * dimY * dimZ;

            // ===== parse TF =====
            TFWrapper tfTemp = JsonUtility.FromJson<TFWrapper>(tfJson);
            if (tfTemp == null) throw new Exception("Failed to parse TF JSON");

            VRDFTransferFunction tf = new VRDFTransferFunction();
            tf.type = tfTemp.type;
            tf.entries = tfTemp.entries ?? new List<VRDFTFEntry>();
            tf.curve   = tfTemp.curve   ?? new List<VRDFTransferFunction.TFPoint>();
            tf.origin  = tfTemp.origin;
            tf.channel_names_hint = tfTemp.channel_names_hint ?? new List<string>();

            // ===== dispatch selon mode =====
            VRDFVolumeData result = new VRDFVolumeData();
            result.meta = meta;
            result.tf = tf;

            if (meta.mode == "anatomy_label_weighted" && meta.channels == 2)
            {
                // RAW = [label(float32), weight(float32)] * voxelCount
                int expectedBytes = voxelCount * 2 * 4;
                if ((int)rawLen != expectedBytes)
                {
                    throw new Exception(
                        $"RAW length mismatch for fused lw: got {rawLen} bytes but expected {expectedBytes}"
                    );
                }

                // on sépare
                result.labelData = new float[voxelCount];
                result.weightData = new float[voxelCount];

                // BlockCopy ne marche pas direct car c'est interleavé.
                // On itère manuellement :
                // rawBytes : [l0,w0,l1,w1,l2,w2,...] little-endian float32
                // On lit via Buffer.BlockCopy en pas de 8? -> pas direct, donc BitConverter.
                // Pour rester safe et clair, on fait un loop.
                // (Si perfs critiques: compute shader/offline unpack. Là on vise la clarté.)
                int stride = 8; // 2 * sizeof(float)
                for (int i = 0; i < voxelCount; i++)
                {
                    int off = i * stride;
                    float lbl = BitConverter.ToSingle(rawBytes, off + 0);
                    float w   = BitConverter.ToSingle(rawBytes, off + 4);
                    result.labelData[i]  = lbl;
                    result.weightData[i] = w;
                }
            }
            else
            {
                // Cas legacy : RAW = 1 float32 par voxel
                int expectedBytes = voxelCount * 4;
                if ((int)rawLen != expectedBytes)
                {
                    throw new Exception(
                        $"RAW length mismatch: got {rawLen} bytes but expected {expectedBytes} ({dimX}x{dimY}x{dimZ} float32)"
                    );
                }

                result.volumeData = new float[voxelCount];
                Buffer.BlockCopy(rawBytes, 0, result.volumeData, 0, expectedBytes);
            }

            return result;
        }
    }

    // Construction des Textures Unity selon le mode
    public static void BuildUnityTextures(VRDFVolumeData data)
    {
        var dim = data.meta.dim;
        int dimX = dim[0];
        int dimY = dim[1];
        int dimZ = dim[2];

        if (data.meta.mode == "anatomy_label_weighted" && data.meta.channels == 2)
        {
            // labelTexture = RFloat (contient les labels discrets genre 0,1,2...)
            Texture3D texLbl = new Texture3D(dimX, dimY, dimZ, TextureFormat.RFloat, false);
            texLbl.wrapMode   = TextureWrapMode.Clamp;
            texLbl.filterMode = FilterMode.Point; // nearest, important pour les labels
            texLbl.SetPixelData(data.labelData, 0);
            texLbl.Apply(false, false);
            data.labelTexture = texLbl;

            // weightTexture = RFloat [0..1]
            Texture3D texW = new Texture3D(dimX, dimY, dimZ, TextureFormat.RFloat, false);
            texW.wrapMode   = TextureWrapMode.Clamp;
            texW.filterMode = FilterMode.Bilinear;
            texW.SetPixelData(data.weightData, 0);
            texW.Apply(false, false);
            data.weightTexture = texW;

            // LUT pour les labels (comme avant, à partir de tf.entries)
            if (data.tf.type == "labelmap")
            {
                BuildLabelmapLUTs(data);
            }
            else
            {
                data.tfLUTTextureSoft = null;
                data.tfLUTTextureHard = null;
            }
        }
        else
        {
            // ancien cas: un seul volumeData -> volumeTexture
            Texture3D tex3d = new Texture3D(dimX, dimY, dimZ, TextureFormat.RFloat, false);
            tex3d.wrapMode   = TextureWrapMode.Clamp;
            tex3d.filterMode = (data.tf != null && data.tf.type == "labelmap")
                                 ? FilterMode.Point
                                 : FilterMode.Bilinear;
            tex3d.SetPixelData(data.volumeData, 0);
            tex3d.Apply(false, false);
            data.volumeTexture = tex3d;

            if (data.tf.type == "labelmap")
            {
                BuildLabelmapLUTs(data);
            }
            else if (data.tf.type == "continuous")
            {
                Texture2D lut = BuildContinuousLUT(data.tf);
                data.tfLUTTextureSoft = lut;
                data.tfLUTTextureHard = null;
            }
            else
            {
                data.tfLUTTextureSoft = null;
                data.tfLUTTextureHard = null;
            }
        }
    }

    private static Texture2D BuildContinuousLUT(VRDFTransferFunction tf)
    {
        if (tf.curve == null || tf.curve.Count == 0)
        {
            Debug.LogWarning("[VRDFLoader] TF curve empty for continuous volume");
            return null;
        }

        int lutSize = tf.curve.Count;
        Texture2D lut = new Texture2D(lutSize, 1, TextureFormat.RGBAFloat, false);
        lut.wrapMode   = TextureWrapMode.Clamp;
        lut.filterMode = FilterMode.Bilinear;

        Color[] pixels = new Color[lutSize];
        for (int i = 0; i < lutSize; i++)
        {
            var p = tf.curve[i];
            float r = (p.color != null && p.color.Length > 0) ? p.color[0] : 0f;
            float g = (p.color != null && p.color.Length > 1) ? p.color[1] : 0f;
            float b = (p.color != null && p.color.Length > 2) ? p.color[2] : 0f;
            float a = p.alpha;
            pixels[i] = new Color(r, g, b, a);
        }

        lut.SetPixels(pixels);
        lut.Apply(false, false);
        return lut;
    }

    private static void BuildLabelmapLUTs(VRDFVolumeData data)
    {
        const int lutSize = 256;

        Color[]   floatPixels = new Color[lutSize];
        Color32[] bytePixels  = new Color32[lutSize];

        for (int i = 0; i < lutSize; i++)
        {
            floatPixels[i] = new Color(0f, 0f, 0f, 0f);
            bytePixels[i]  = new Color32(0, 0, 0, 0);
        }

        if (data.tf.entries != null)
        {
            foreach (var e in data.tf.entries)
            {
                int idx = Mathf.Clamp(Mathf.RoundToInt(e.label), 0, lutSize - 1);

                float r = (e.color != null && e.color.Length > 0) ? e.color[0] : 0f;
                float g = (e.color != null && e.color.Length > 1) ? e.color[1] : 0f;
                float b = (e.color != null && e.color.Length > 2) ? e.color[2] : 0f;
                float a = e.alpha;

                floatPixels[idx] = new Color(r, g, b, a);

                byte R = (byte)Mathf.Clamp(Mathf.RoundToInt(r * 255f), 0, 255);
                byte G = (byte)Mathf.Clamp(Mathf.RoundToInt(g * 255f), 0, 255);
                byte B = (byte)Mathf.Clamp(Mathf.RoundToInt(b * 255f), 0, 255);
                byte A = (byte)Mathf.Clamp(Mathf.RoundToInt(a * 255f), 0, 255);
                bytePixels[idx] = new Color32(R, G, B, A);
            }
        }

        Texture2D lutSoft = new Texture2D(lutSize, 1, TextureFormat.RGBAFloat, false);
        lutSoft.wrapMode   = TextureWrapMode.Clamp;
        lutSoft.filterMode = FilterMode.Bilinear;
        lutSoft.SetPixels(floatPixels);
        lutSoft.Apply(false, false);

        Texture2D lutHard = new Texture2D(lutSize, 1, TextureFormat.RGBA32, false);
        lutHard.wrapMode   = TextureWrapMode.Clamp;
        lutHard.filterMode = FilterMode.Point;
        lutHard.SetPixels32(bytePixels);
        lutHard.Apply(false, false);

        data.tfLUTTextureSoft = lutSoft;
        data.tfLUTTextureHard = lutHard;
    }
}
