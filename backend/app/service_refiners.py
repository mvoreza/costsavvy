"""
Service Refiners - Fallback definitions for service clarification flows.

Refiners help narrow down generic service queries (like "MRI") to specific 
billable CPT codes (like "MRI Brain without contrast - CPT 70551").

This file provides a Python fallback when the database refiners table is empty or unavailable.
"""

def refiners_registry() -> dict:
    """
    Returns the fallback refiners configuration.
    
    Structure:
    {
        "refiners": [
            {
                "id": "unique-id",
                "match": {"keywords": ["keyword1", "keyword2"]},
                "prompt": "Question to ask user",
                "choices": [
                    {"key": "1", "label": "Option 1", "code_type": "CPT", "code": "12345"},
                    ...
                ],
                "preview_code_type": "CPT",  # Optional: default if user skips
                "preview_code": "12345"
            },
            ...
        ]
    }
    """
    return {
        "refiners": [
            # MRI Refiners
            {
                "id": "mri-general",
                "match": {"keywords": ["mri", "magnetic resonance"]},
                "prompt": "Which body area is the MRI for?",
                "choices": [
                    {"key": "1", "label": "Brain/Head", "code_type": "CPT", "code": "70551"},
                    {"key": "2", "label": "Cervical Spine (Neck)", "code_type": "CPT", "code": "72141"},
                    {"key": "3", "label": "Thoracic Spine (Mid-back)", "code_type": "CPT", "code": "72146"},
                    {"key": "4", "label": "Lumbar Spine (Lower back)", "code_type": "CPT", "code": "72148"},
                    {"key": "5", "label": "Knee", "code_type": "CPT", "code": "73721"},
                    {"key": "6", "label": "Shoulder", "code_type": "CPT", "code": "73221"},
                    {"key": "7", "label": "Hip", "code_type": "CPT", "code": "73721"},
                    {"key": "8", "label": "Abdomen", "code_type": "CPT", "code": "74181"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "70551"
            },
            
            # CT Scan Refiners
            {
                "id": "ct-general",
                "match": {"keywords": ["ct scan", "cat scan", "computed tomography"]},
                "prompt": "Which body area is the CT scan for?",
                "choices": [
                    {"key": "1", "label": "Head/Brain", "code_type": "CPT", "code": "70450"},
                    {"key": "2", "label": "Chest", "code_type": "CPT", "code": "71250"},
                    {"key": "3", "label": "Abdomen", "code_type": "CPT", "code": "74150"},
                    {"key": "4", "label": "Abdomen & Pelvis", "code_type": "CPT", "code": "74176"},
                    {"key": "5", "label": "Cervical Spine (Neck)", "code_type": "CPT", "code": "72125"},
                    {"key": "6", "label": "Lumbar Spine (Lower back)", "code_type": "CPT", "code": "72131"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "70450"
            },
            
            # X-Ray Refiners
            {
                "id": "xray-general",
                "match": {"keywords": ["x-ray", "xray", "x ray", "radiograph"]},
                "prompt": "Which body area is the X-ray for?",
                "choices": [
                    {"key": "1", "label": "Chest (1 view)", "code_type": "CPT", "code": "71045"},
                    {"key": "2", "label": "Chest (2 views)", "code_type": "CPT", "code": "71046"},
                    {"key": "3", "label": "Knee", "code_type": "CPT", "code": "73562"},
                    {"key": "4", "label": "Shoulder", "code_type": "CPT", "code": "73030"},
                    {"key": "5", "label": "Hip", "code_type": "CPT", "code": "73502"},
                    {"key": "6", "label": "Ankle", "code_type": "CPT", "code": "73610"},
                    {"key": "7", "label": "Foot", "code_type": "CPT", "code": "73630"},
                    {"key": "8", "label": "Hand", "code_type": "CPT", "code": "73130"},
                    {"key": "9", "label": "Wrist", "code_type": "CPT", "code": "73110"},
                    {"key": "10", "label": "Spine (Lumbar)", "code_type": "CPT", "code": "72100"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "71046"
            },
            
            # Ultrasound Refiners
            {
                "id": "ultrasound-general",
                "match": {"keywords": ["ultrasound", "sonogram", "echo"]},
                "prompt": "Which type of ultrasound?",
                "choices": [
                    {"key": "1", "label": "Abdominal (Complete)", "code_type": "CPT", "code": "76700"},
                    {"key": "2", "label": "Pelvic", "code_type": "CPT", "code": "76856"},
                    {"key": "3", "label": "Thyroid", "code_type": "CPT", "code": "76536"},
                    {"key": "4", "label": "Breast", "code_type": "CPT", "code": "76641"},
                    {"key": "5", "label": "Echocardiogram (Heart)", "code_type": "CPT", "code": "93306"},
                    {"key": "6", "label": "Pregnancy/OB", "code_type": "CPT", "code": "76801"},
                    {"key": "7", "label": "Carotid (Neck vessels)", "code_type": "CPT", "code": "93880"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "76700"
            },
            
            # Colonoscopy Refiners
            {
                "id": "colonoscopy",
                "match": {"keywords": ["colonoscopy"]},
                "prompt": "Is this a screening (preventive) or diagnostic colonoscopy?",
                "choices": [
                    {"key": "1", "label": "Screening (no symptoms, routine)", "code_type": "CPT", "code": "45378"},
                    {"key": "2", "label": "Diagnostic (symptoms or follow-up)", "code_type": "CPT", "code": "45380"},
                    {"key": "3", "label": "With biopsy", "code_type": "CPT", "code": "45380"},
                    {"key": "4", "label": "With polyp removal", "code_type": "CPT", "code": "45385"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "45378"
            },
            
            # Mammogram Refiners
            {
                "id": "mammogram",
                "match": {"keywords": ["mammogram", "mammography", "breast imaging"]},
                "prompt": "Is this a screening or diagnostic mammogram?",
                "choices": [
                    {"key": "1", "label": "Screening (routine, no symptoms)", "code_type": "CPT", "code": "77067"},
                    {"key": "2", "label": "Diagnostic (symptoms or follow-up)", "code_type": "CPT", "code": "77066"},
                    {"key": "3", "label": "3D Mammogram (Tomosynthesis)", "code_type": "CPT", "code": "77063"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "77067"
            },
            
            # Lab Tests
            {
                "id": "lab-blood",
                "match": {"keywords": ["blood test", "lab test", "blood work", "labs"]},
                "prompt": "Which type of lab test?",
                "choices": [
                    {"key": "1", "label": "Complete Blood Count (CBC)", "code_type": "CPT", "code": "85025"},
                    {"key": "2", "label": "Basic Metabolic Panel", "code_type": "CPT", "code": "80048"},
                    {"key": "3", "label": "Comprehensive Metabolic Panel", "code_type": "CPT", "code": "80053"},
                    {"key": "4", "label": "Lipid Panel (Cholesterol)", "code_type": "CPT", "code": "80061"},
                    {"key": "5", "label": "Hemoglobin A1C (Diabetes)", "code_type": "CPT", "code": "83036"},
                    {"key": "6", "label": "Thyroid Panel (TSH)", "code_type": "CPT", "code": "84443"},
                    {"key": "7", "label": "Urinalysis", "code_type": "CPT", "code": "81003"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "80053"
            },
            
            # Office Visits
            {
                "id": "office-visit",
                "match": {"keywords": ["office visit", "doctor visit", "checkup", "physical"]},
                "prompt": "What type of visit?",
                "choices": [
                    {"key": "1", "label": "New patient visit", "code_type": "CPT", "code": "99203"},
                    {"key": "2", "label": "Established patient visit", "code_type": "CPT", "code": "99213"},
                    {"key": "3", "label": "Annual physical/wellness", "code_type": "CPT", "code": "99395"},
                    {"key": "4", "label": "Specialist consultation", "code_type": "CPT", "code": "99243"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "99213"
            },
            
            # Emergency Room
            {
                "id": "emergency",
                "match": {"keywords": ["emergency room", "er visit", "emergency visit", "ed visit"]},
                "prompt": "What level of emergency care?",
                "choices": [
                    {"key": "1", "label": "Low complexity", "code_type": "CPT", "code": "99281"},
                    {"key": "2", "label": "Moderate complexity", "code_type": "CPT", "code": "99283"},
                    {"key": "3", "label": "High complexity", "code_type": "CPT", "code": "99284"},
                    {"key": "4", "label": "Critical/Life-threatening", "code_type": "CPT", "code": "99285"},
                ],
                "preview_code_type": "CPT",
                "preview_code": "99283"
            },
        ]
    }
