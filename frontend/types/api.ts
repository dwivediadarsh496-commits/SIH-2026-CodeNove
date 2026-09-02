export type HealthResponse = {
  status: "ok" | "initializing" | string;
  model_ready: boolean;
  device: string;
  cuda_available: boolean;
  checkpoint_loaded: boolean;
  service: string;
  version: string;
};

export type ModelInfoResponse = {
  model_name: string;
  architecture: string;
  in_channels: number;
  out_channels: number;
  scale_factor: number;
  trainable_parameters: number;
  device: string;
  cuda_available: boolean;
  checkpoint_loaded: boolean;
  checkpoint_info: Record<string, unknown>;
  input_bands: string[];
  input_resolution: string;
  target_representation: string;
  normalization: string;
  mc_samples: number;
  tile_size: number;
  tile_overlap: number;
};

export type SuperResolutionResult = {
  success: boolean;
  session_id: string;
  input: {
    filename: string;
    width: number;
    height: number;
    bands: number;
    format: string;
    crs: string | null;
    resolution: string;
  };
  model: {
    name: string;
    architecture: string;
    scale_factor: number;
    device: string;
    checkpoint_loaded: boolean;
    parameters: number;
  };
  output: {
    width: number;
    height: number;
    bands: number;
    sr_image_b64: string;
    uncertainty_map_b64: string;
    tiled_inference: boolean;
    inference_time_s: number;
  };
  metrics: {
    psnr: number | null;
    ssim: number | null;
    rmse: number | null;
    sam: number | null;
    ergas: number | null;
    available: boolean;
    reason?: string;
  };
  downloads: {
    sr_geotiff: string;
    sr_preview: string;
    uncertainty: string;
  };
};

export type LocalImageMeta = {
  filename: string;
  width: number | null;
  height: number | null;
  sizeBytes: number;
  format: string;
  acquisitionTime: string;
  previewUrl: string | null;
};
