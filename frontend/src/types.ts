export interface NLPParams {
  features: {
    enhancement: boolean;
    makeup: boolean;
    reshape: boolean;
    smoothing: boolean;
  };
  enhancement_params: Record<string, any>;
  makeup_config: Record<string, any>;
  reshape_config: Record<string, any>;
  beauty_config: Record<string, any>;
}
