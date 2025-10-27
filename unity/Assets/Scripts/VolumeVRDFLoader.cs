using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

[Serializable]
public class VRDFMeta
{
    public int[] dim;                  // [dimX, dimY, dimZ]
    public float[] spacing_mm;         // [sx, sy, sz]
    public string dtype;               // "float32"
    public float[] intensity_range;    // [min, max]
    public float[,] affine;            // 4x4 (rebuilt below)
    public string mode;                // "labelmap"|"continuous"|...
    public string endianness;
    public string order;
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
    public float[] volumeData;     // length = dimX*dimY*dimZ

    // Unity runtime textures
    public Texture3D volumeTexture;   // RFloat
    public Texture2D tfLUTTexture;    // RGBA32 256x1 (labelmap) or gradient
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
            // ---- header ----
            byte[] magicBytes = br.ReadBytes(8);
            if (magicBytes.Length != 8) throw new Exception("File too short (no magic)");
            string magic = Encoding.ASCII.GetString(magicBytes);
            if (magic != "VRDF0001")
                throw new Exception($"Bad magic '{magic}', expected 'VRDF0001'.");

            ulong totalSize = VRDFBinaryUtil.ReadUInt64LE(br);

            // ---- meta json ----
            ulong metaLen = VRDFBinaryUtil.ReadUInt64LE(br);
            byte[] metaBytes = br.ReadBytes((int)metaLen);
            if ((ulong)metaBytes.Length != metaLen)
                throw new Exception("EOF in meta block");
            string metaJson = Encoding.UTF8.GetString(metaBytes);

            // ---- tf json ----
            ulong tfLen = VRDFBinaryUtil.ReadUInt64LE(br);
            byte[] tfBytes = br.ReadBytes((int)tfLen);
            if ((ulong)tfBytes.Length != tfLen)
                throw new Exception("EOF in TF block");
            string tfJson = Encoding.UTF8.GetString(tfBytes);

            // ---- raw data ----
            ulong rawLen = VRDFBinaryUtil.ReadUInt64LE(br);
            byte[] rawBytes = br.ReadBytes((int)rawLen);
            if ((ulong)rawBytes.Length != rawLen)
                throw new Exception("EOF in RAW block");

            // parse meta
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
            int expectedBytes = voxelCount * 4;
            if ((int)rawLen != expectedBytes)
            {
                throw new Exception(
                    $"RAW length mismatch: got {rawLen} bytes but expected {expectedBytes} ({dimX}x{dimY}x{dimZ} float32)"
                );
            }

            // parse tf
            TFWrapper tfTemp = JsonUtility.FromJson<TFWrapper>(tfJson);
            if (tfTemp == null) throw new Exception("Failed to parse TF JSON");

            VRDFTransferFunction tf = new VRDFTransferFunction();
            tf.type = tfTemp.type;
            tf.entries = tfTemp.entries ?? new List<VRDFTFEntry>();
            tf.curve   = tfTemp.curve   ?? new List<VRDFTransferFunction.TFPoint>();
            tf.origin  = tfTemp.origin;
            tf.channel_names_hint = tfTemp.channel_names_hint ?? new List<string>();

            // copy rawBytes -> float[]
            float[] volumeData = new float[voxelCount];
            Buffer.BlockCopy(rawBytes, 0, volumeData, 0, expectedBytes);

            VRDFVolumeData result = new VRDFVolumeData();
            result.meta = meta;
            result.tf = tf;
            result.volumeData = volumeData;

            return result;
        }
    }

    public static void BuildUnityTextures(VRDFVolumeData data)
    {
        var dim = data.meta.dim;
        int dimX = dim[0];
        int dimY = dim[1];
        int dimZ = dim[2];

        // -------- Volume Texture3D (RFloat) --------
        // Unity's Texture3D.SetPixelData(float[]) expects index = x + width*(y + height*z)
        // Our volumeData is stored exactly that way in write_vrdf(), so we can upload directly.
        Texture3D tex3d = new Texture3D(dimX, dimY, dimZ, TextureFormat.RFloat, false);
        tex3d.wrapMode = TextureWrapMode.Clamp;
        tex3d.filterMode = FilterMode.Bilinear;
        tex3d.SetPixelData(data.volumeData, 0);
        tex3d.Apply(false, false);

        data.volumeTexture = tex3d;

        // -------- Transfer Function LUT --------
        if (data.tf.type == "labelmap")
        {
            // Shader assumes a 256-wide LUT indexed by label value.
            // So we ALWAYS build 256x1, even if we have only 5 classes.
            const int lutSize = 256;
            Texture2D lutTex = new Texture2D(lutSize, 1, TextureFormat.RGBA32, false);
            lutTex.wrapMode = TextureWrapMode.Clamp;
            lutTex.filterMode = FilterMode.Point;

            Color32[] lutColors = new Color32[lutSize];
            for (int i = 0; i < lutSize; i++)
                lutColors[i] = new Color32(0,0,0,0);

            foreach (var e in data.tf.entries)
            {
                int idx = Mathf.RoundToInt(e.label);
                if (idx < 0 || idx >= lutSize)
                    continue;

                float r = (e.color != null && e.color.Length > 0) ? e.color[0] : 1f;
                float g = (e.color != null && e.color.Length > 1) ? e.color[1] : 1f;
                float b = (e.color != null && e.color.Length > 2) ? e.color[2] : 1f;
                float a = e.alpha;

                byte R = (byte)Mathf.Clamp(Mathf.RoundToInt(r * 255f), 0,255);
                byte G = (byte)Mathf.Clamp(Mathf.RoundToInt(g * 255f), 0,255);
                byte B = (byte)Mathf.Clamp(Mathf.RoundToInt(b * 255f), 0,255);
                byte A = (byte)Mathf.Clamp(Mathf.RoundToInt(a * 255f), 0,255);

                lutColors[idx] = new Color32(R,G,B,A);
            }

            lutTex.SetPixels32(lutColors);
            lutTex.Apply(false, false);
            data.tfLUTTexture = lutTex;
        }
        else if (data.tf.type == "continuous")
        {
            // TODO: build a gradient LUT by sampling tf.curve into 256 steps
            // For now we can fallback to a dummy 256x1 white alpha ramp if you need visibility
            // or leave null and handle in shader.
            data.tfLUTTexture = null;
        }
        else
        {
            data.tfLUTTexture = null;
        }
    }
}
