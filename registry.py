# Available watermarking and detection methods
BASELINE_METHODS = {
    'watermarking': {
        'SimpleWatermark': 'methods.watermarking.SimpleWatermark',
        'InvisibleWatermark': 'methods.watermarking.InvisibleWatermark',
    },
    'detection': {
        'NoDetection': 'methods.detection.NoDetection',
        'BasicDetector': 'methods.detection.BasicDetector',
    },
    'attacks': {
        'GaussianNoiseAttack': 'attacks.gaussian_noise_attack.GaussianNoiseAttack',
        'AdversarialPatchAttack': 'attacks.adversarial_patch_attack.AdversarialPatchAttack',
    }
}

# Baseline team configurations
BASELINE_TEAMS = {
    'BaseBlueTeam': {
        'type': 'blue',
        'watermark_method': 'SimpleWatermark',
        'detection_method': 'BasicDetector'
    },
    'BaseRedTeam': {
        'type': 'red',
        'attack_method': 'GaussianNoiseAttack'
    },
    'NoWatermarkTeam': {
        'type': 'blue',
        'watermark_method': None,
        'detection_method': None
    },
    'NoAttackTeam': {
        'type': 'red',
        'attack_method': None
    }
}

# Students can register their teams here
STUDENT_TEAMS = {
    'AdvancedBlueTeam': {
        'type': 'blue',
        'watermark_method': 'InvisibleWatermark',
        'detection_method': 'AdvancedDetector'
    },
    'AdvancedRedTeam1': {
        'type': 'red',
        'attack_method': 'AdversarialPatchAttack'
    },
    # Add more student teams as needed
}