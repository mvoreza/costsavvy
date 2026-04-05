# app/service_refiners.py
from __future__ import annotations

from typing import Dict, Any, List, Optional


def refiners_registry() -> Dict[str, Any]:
    """
    Python fallback registry for service refiners.
    Shape mirrors what the DB loader returns:
      { "version": 1, "refiners": [ ... ] }
    """
    return {
        "version": 1,
        "refiners": [
            # 1) Colonoscopy
            {
                "id": "colonoscopy",
                "title": "Colonoscopy",
                "match": {"keywords": ["colonoscopy"]},
                "require_choice_before_pricing": False,
                "preview_code": {"code_type": "CPT", "code": "45378"},
                "question": "To make the price more exact, which best matches what you need?",
                "choices": [
                    {"key": "1", "label": "Standard colonoscopy (no biopsy or polyp removal)", "code_type": "CPT", "code": "45378"},
                    {"key": "2", "label": "Colonoscopy with biopsy", "code_type": "CPT", "code": "45380"},
                    {"key": "3", "label": "Colonoscopy with polyp removal", "code_type": "CPT", "code": "45385"},
                ],
            },

            # 2) Upper endoscopy (EGD)
            {
                "id": "upper_endoscopy_egd",
                "title": "Upper endoscopy (EGD)",
                "match": {"keywords": ["egd", "upper endoscopy", "upper gi endoscopy", "esophagogastroduodenoscopy"]},
                "require_choice_before_pricing": False,
                "preview_code": {"code_type": "CPT", "code": "43235"},
                "question": "Which type of upper endoscopy is closest?",
                "choices": [
                    {"key": "1", "label": "EGD diagnostic (no biopsy)", "code_type": "CPT", "code": "43235"},
                    {"key": "2", "label": "EGD with biopsy", "code_type": "CPT", "code": "43239"},
                ],
            },

            # 3) MRI Brain
            {
                "id": "mri_brain",
                "title": "MRI Brain",
                "match": {"keywords": ["mri brain", "brain mri"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "MRI brain prices differ by contrast. Which one do you need?",
                "choices": [
                    {"key": "1", "label": "Without contrast", "code_type": "CPT", "code": "70551"},
                    {"key": "2", "label": "With contrast", "code_type": "CPT", "code": "70552"},
                    {"key": "3", "label": "With and without contrast", "code_type": "CPT", "code": "70553"},
                ],
            },

            # 4) MRI Lumbar Spine
            {
                "id": "mri_lumbar_spine",
                "title": "MRI Lumbar Spine",
                "match": {"keywords": ["mri lumbar", "lumbar mri", "mri low back", "mri spine lumbar"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "MRI lumbar spine prices differ by contrast. Which one do you need?",
                "choices": [
                    {"key": "1", "label": "Without contrast", "code_type": "CPT", "code": "72148"},
                    {"key": "2", "label": "With contrast", "code_type": "CPT", "code": "72149"},
                    {"key": "3", "label": "With and without contrast", "code_type": "CPT", "code": "72158"},
                ],
            },

            # 5) CT Head
            {
                "id": "ct_head",
                "title": "CT Head/Brain",
                "match": {"keywords": ["ct head", "head ct", "ct brain", "brain ct"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "CT head prices differ by contrast. Which one do you need?",
                "choices": [
                    {"key": "1", "label": "Without contrast", "code_type": "CPT", "code": "70450"},
                    {"key": "2", "label": "With contrast", "code_type": "CPT", "code": "70460"},
                    {"key": "3", "label": "With and without contrast", "code_type": "CPT", "code": "70470"},
                ],
            },

            # 6) CT Abdomen/Pelvis
            {
                "id": "ct_abdomen_pelvis",
                "title": "CT Abdomen & Pelvis",
                "match": {"keywords": ["ct abdomen pelvis", "ct abdomen and pelvis", "ct a/p", "ct abd pelvis"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "CT abdomen/pelvis prices differ by contrast. Which one do you need?",
                "choices": [
                    {"key": "1", "label": "Without contrast", "code_type": "CPT", "code": "74176"},
                    {"key": "2", "label": "With contrast", "code_type": "CPT", "code": "74177"},
                    {"key": "3", "label": "With and without contrast", "code_type": "CPT", "code": "74178"},
                ],
            },

            # 7) Chest X-ray
            {
                "id": "xray_chest",
                "title": "Chest X-ray",
                "match": {"keywords": ["chest x-ray", "chest xray"]},
                "require_choice_before_pricing": False,
                "preview_code": {"code_type": "CPT", "code": "71046"},
                "question": "Chest X-ray prices vary by number of views. Which is closest?",
                "choices": [
                    {"key": "1", "label": "1 view", "code_type": "CPT", "code": "71045"},
                    {"key": "2", "label": "2 views (most common)", "code_type": "CPT", "code": "71046"},
                    {"key": "3", "label": "3–4 views", "code_type": "CPT", "code": "71047"},
                ],
            },

            # 8) Knee X-ray
            {
                "id": "xray_knee",
                "title": "Knee X-ray",
                "match": {"keywords": ["knee x-ray", "knee xray"]},
                "require_choice_before_pricing": False,
                "preview_code": {"code_type": "CPT", "code": "73562"},
                "question": "Knee X-ray prices vary by number of views. Which is closest?",
                "choices": [
                    {"key": "1", "label": "1–2 views", "code_type": "CPT", "code": "73560"},
                    {"key": "2", "label": "3 views (most common)", "code_type": "CPT", "code": "73562"},
                    {"key": "3", "label": "4+ views", "code_type": "CPT", "code": "73564"},
                ],
            },

            # 9) Mammogram
            {
                "id": "mammogram",
                "title": "Mammogram",
                "match": {"keywords": ["mammogram", "mammo"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "Is this a screening or diagnostic mammogram (diagnostic often costs more)?",
                "choices": [
                    {"key": "1", "label": "Screening mammogram", "code_type": "CPT", "code": "77067"},
                    {"key": "2", "label": "Diagnostic mammogram", "code_type": "CPT", "code": "77066"},
                ],
            },

            # 10) Ultrasound Abdomen
            {
                "id": "ultrasound_abdomen",
                "title": "Abdominal ultrasound",
                "match": {"keywords": ["abdominal ultrasound", "ultrasound abdomen", "ruq ultrasound", "right upper quadrant ultrasound"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "Which abdominal ultrasound is it?",
                "choices": [
                    {"key": "1", "label": "Limited (e.g., RUQ)", "code_type": "CPT", "code": "76705"},
                    {"key": "2", "label": "Complete abdomen", "code_type": "CPT", "code": "76700"},
                ],
            },

            # 11) Ultrasound Pelvis
            {
                "id": "ultrasound_pelvis",
                "title": "Pelvic ultrasound",
                "match": {"keywords": ["pelvic ultrasound", "ultrasound pelvis"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "Which pelvic ultrasound is it?",
                "choices": [
                    {"key": "1", "label": "Pelvic ultrasound (non-obstetric), complete", "code_type": "CPT", "code": "76856"},
                    {"key": "2", "label": "Transvaginal ultrasound", "code_type": "CPT", "code": "76830"},
                ],
            },

            # 12) Echocardiogram (TTE)
            {
                "id": "echo_tte",
                "title": "Echocardiogram (TTE)",
                "match": {"keywords": ["echocardiogram", "echo", "tte"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "Which echocardiogram is it?",
                "choices": [
                    {"key": "1", "label": "TTE, complete (common)", "code_type": "CPT", "code": "93306"},
                    {"key": "2", "label": "TTE, limited/follow-up", "code_type": "CPT", "code": "93308"},
                ],
            },

            # 13) EKG/ECG
            {
                "id": "ekg_ecg",
                "title": "EKG/ECG",
                "match": {"keywords": ["ekg", "ecg", "electrocardiogram"]},
                "require_choice_before_pricing": False,
                "preview_code": {"code_type": "CPT", "code": "93000"},
                "question": "Which best matches the EKG service?",
                "choices": [
                    {"key": "1", "label": "Tracing + interpretation (common office)", "code_type": "CPT", "code": "93000"},
                    {"key": "2", "label": "Tracing only", "code_type": "CPT", "code": "93005"},
                    {"key": "3", "label": "Interpretation only", "code_type": "CPT", "code": "93010"},
                ],
            },

            # 14) CBC
            {
                "id": "lab_cbc",
                "title": "CBC blood test",
                "match": {"keywords": ["cbc", "complete blood count"]},
                "require_choice_before_pricing": False,
                "preview_code": {"code_type": "CPT", "code": "85025"},
                "question": "Do you need a CBC with differential (most common) or without differential?",
                "choices": [
                    {"key": "1", "label": "CBC with automated differential (most common)", "code_type": "CPT", "code": "85025"},
                    {"key": "2", "label": "CBC without differential", "code_type": "CPT", "code": "85027"},
                ],
            },

            # 15) CMP vs BMP
            {
                "id": "lab_cmp_bmp",
                "title": "Metabolic panel",
                "match": {"keywords": ["cmp", "comprehensive metabolic panel", "bmp", "basic metabolic panel", "metabolic panel"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "Which metabolic panel do you need?",
                "choices": [
                    {"key": "1", "label": "CMP (comprehensive metabolic panel)", "code_type": "CPT", "code": "80053"},
                    {"key": "2", "label": "BMP (basic metabolic panel)", "code_type": "CPT", "code": "80048"},
                ],
            },

            # 16) A1c
            {
                "id": "lab_a1c",
                "title": "Hemoglobin A1c",
                "match": {"keywords": ["a1c", "hemoglobin a1c"]},
                "require_choice_before_pricing": False,
                "preview_code": {"code_type": "CPT", "code": "83036"},
                "question": "Confirming: you mean Hemoglobin A1c?",
                "choices": [{"key": "1", "label": "Yes, A1c", "code_type": "CPT", "code": "83036"}],
            },

            # 17) Lipid panel
            {
                "id": "lab_lipid",
                "title": "Lipid panel",
                "match": {"keywords": ["lipid panel", "cholesterol test"]},
                "require_choice_before_pricing": False,
                "preview_code": {"code_type": "CPT", "code": "80061"},
                "question": "Confirming: you mean a lipid panel (cholesterol test)?",
                "choices": [{"key": "1", "label": "Yes, lipid panel", "code_type": "CPT", "code": "80061"}],
            },

            # 18) Office visit (new vs established)
            {
                "id": "office_visit",
                "title": "Office visit (E/M)",
                "match": {"keywords": ["office visit", "doctor visit", "primary care visit", "clinic visit"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "For office visit pricing, are you a new patient or an established patient?",
                "choices": [
                    {"key": "1", "label": "New patient (typical level 3)", "code_type": "CPT", "code": "99203"},
                    {"key": "2", "label": "Established patient (typical level 3)", "code_type": "CPT", "code": "99213"},
                ],
            },

            # 19) CT Chest
            {
                "id": "ct_chest",
                "title": "CT Chest",
                "match": {"keywords": ["ct chest", "chest ct"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "CT chest prices differ by contrast. Which one do you need?",
                "choices": [
                    {"key": "1", "label": "Without contrast", "code_type": "CPT", "code": "71250"},
                    {"key": "2", "label": "With contrast", "code_type": "CPT", "code": "71260"},
                    {"key": "3", "label": "With and without contrast", "code_type": "CPT", "code": "71270"},
                ],
            },

            # 20) MRI Knee
            {
                "id": "mri_knee",
                "title": "MRI Knee",
                "match": {"keywords": ["mri knee", "knee mri"]},
                "require_choice_before_pricing": True,
                "preview_code": None,
                "question": "MRI knee prices differ by contrast. Which one do you need?",
                "choices": [
                    {"key": "1", "label": "Without contrast", "code_type": "CPT", "code": "73721"},
                    {"key": "2", "label": "With contrast", "code_type": "CPT", "code": "73722"},
                    {"key": "3", "label": "With and without contrast", "code_type": "CPT", "code": "73723"},
                ],
            },
        ],
    }
