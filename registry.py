# registry.py

BASELINE_METHODS = {
    'watermarking': {
        'BaseWatermarkedDiffusionPipeline': 'methods.watermarked_diffusion_pipeline.BaseWatermarkedDiffusionPipeline',
    },
    'attacks': {
        'NoAttack': 'attacks.base_attack.NoAttack',
        # Add other attack methods here
    }
}

BASELINE_TEAMS = {
    'NoWatermarkTeam': {
        'type': 'blue',
        'watermark_method': 'BaseWatermarkedDiffusionPipeline'
    },
    'NoAttackTeam': {
        'type': 'red',
        'attack_method': 'NoAttack'
    },
    # Add other baseline teams here
}

STUDENT_TEAMS = {
    # Register your teams here
}
