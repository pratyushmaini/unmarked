# registry.py

BASELINE_METHODS = {
    "watermarking": {
        "BaseWatermarkedDiffusionPipeline": "methods.watermarked_diffusion_pipeline.BaseWatermarkedDiffusionPipeline",
        "OutputPixelWatermarking": "methods.output_pixel_watermarking.OutputPixelWatermarking",
        "ZodiacWatermarking": "methods.zodiac_watermarking.ZodiacWatermarkedPipeline"
    },
    "attacks": {
        "NoAttack": "attacks.base_attack.NoAttack",
        # Add other attack methods here
        "DistortionAttack": "attacks.distortion_attack.DistortionAttack",
        "ChainedDistortionAttack": "attacks.distortion_attack.ChainedDistortionAttack",
    },
}

BASELINE_TEAMS = {
    "NoWatermarkTeam": {
        "type": "blue",
        "watermark_method": "BaseWatermarkedDiffusionPipeline",
    },
    "BaseBlueTeam": {"type": "blue", "watermark_method": "OutputPixelWatermarking"},
    "NoAttackTeam": {"type": "red", "attack_method": "NoAttack"},
    "ZodiacWatermarkTeam": {"type": "blue", "watermark_method": "ZodiacWatermarking"}
    # Add other baseline teams here
}

STUDENT_TEAMS = {
    "OutputPixelTeam": {"type": "blue", "watermark_method": "OutputPixelWatermarking"},
    "DistortionTeam": {
        "type": "red",
        "attack_method": "DistortionAttack",
        "distortion_type": "noise",
    },
    "ChainDistortionTeam": {
        "type": "red",
        "attack_method": "ChainedDistortionAttack",
        "distortion_types": ["noise", "compression"],
    },
    # Register your teams here
    "ZodiacWatermarkTeam": {
        "type": "blue",
        "watermark_method": "ZodiacWatermarking"
    }
}
