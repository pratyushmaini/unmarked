# registry.py

BASELINE_METHODS = {
    'watermarking': {
        'BaseWatermarkedDiffusionPipeline': 'methods.watermarked_diffusion_pipeline.BaseWatermarkedDiffusionPipeline',
        'OutputPixelWatermarking': 'methods.output_pixel_watermarking.OutputPixelWatermarking',
    },
    'attacks': {
        'NoAttack': 'attacks.base_attack.NoAttack',
        # Add other attack methods here
    }
}

BASELINE_TEAMS = {
    'NoWatermarkTeam': {
        'type': 'blue',
        'watermark_method': 'NoWatermarkPipeline'
    },
    'BaseBlueTeam': {
        'type': 'blue',
        'watermark_method': 'OutputPixelWatermarking'
    },
    'NoAttackTeam': {
        'type': 'red',
        'attack_method': 'NoAttack'
    },
    # Add other baseline teams here
}

STUDENT_TEAMS = {
    'OutputPixelTeam': {
        'type': 'blue',
        'watermark_method': 'OutputPixelWatermarking'
    },
    # Register your teams here
}
