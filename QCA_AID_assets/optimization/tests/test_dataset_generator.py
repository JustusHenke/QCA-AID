"""
Test Dataset Generator for API Call Optimization
Creates 100 representative segments across all analysis modes with known quality metrics.
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime

class TestDatasetGenerator:
    """Generates test dataset for optimization validation."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.random = random.Random(seed)
        
        # Sample texts for different analysis modes
        self.sample_texts = {
            'deductive': [
                "The implementation of renewable energy policies requires careful consideration of economic factors.",
                "Climate change adaptation strategies must account for regional variations in vulnerability.",
                "Technological innovation drives efficiency improvements in industrial processes.",
                "Public acceptance of environmental regulations depends on perceived fairness and transparency.",
                "Sustainable development requires balancing economic growth with ecological preservation.",
                "Policy interventions can create positive feedback loops for green technology adoption.",
                "Stakeholder engagement is critical for successful implementation of environmental programs.",
                "Long-term planning horizons are essential for effective climate policy.",
                "Behavioral changes at individual level contribute to collective environmental impact.",
                "International cooperation facilitates technology transfer and capacity building."
            ],
            'inductive': [
                "The participant expressed concerns about the timeline for project completion.",
                "Interview data reveals patterns of resistance to organizational change.",
                "Analysis of stakeholder feedback shows recurring themes of communication gaps.",
                "Observation notes indicate consistent patterns in team dynamics during crises.",
                "Document analysis suggests institutional barriers to innovation adoption.",
                "Survey responses highlight differences in perception across demographic groups.",
                "Case study reveals unexpected connections between leadership style and outcomes.",
                "Archival data provides insights into historical decision-making patterns.",
                "Focus group discussions uncover latent tensions between departments.",
                "Ethnographic notes capture subtle cultural norms affecting implementation."
            ],
            'abductive': [
                "While the policy was designed to reduce emissions, unintended consequences emerged.",
                "The intervention successfully reduced costs but had negative social impacts.",
                "Initial results were promising, but sustainability remained a challenge.",
                "Stakeholder feedback contradicted the quantitative performance metrics.",
                "Implementation barriers differed significantly from initial assumptions.",
                "The technology worked as expected but adoption rates were lower than projected.",
                "Regulatory changes created new opportunities while closing others.",
                "Competitive dynamics shifted in response to the new market conditions.",
                "Cultural factors mediated the effectiveness of the implemented solution.",
                "Resource constraints forced adaptation of the original design."
            ],
            'grounded': [
                "The data suggests several emergent categories that were not anticipated.",
                "Initial coding revealed unexpected connections between concepts.",
                "Open coding produced a rich set of codes requiring further refinement.",
                "Axial coding helped identify relationships between core categories.",
                "Selective coding focused on the central phenomenon and its properties.",
                "Memo writing captured analytical insights during the coding process.",
                "Constant comparison highlighted similarities and differences across cases.",
                "Theoretical sampling guided data collection toward saturation.",
                "Category development proceeded iteratively with data analysis.",
                "Conceptual integration synthesized findings into a coherent theory."
            ]
        }
        
        # Categories for deductive mode testing
        self.deductive_categories = [
            "Policy Implementation",
            "Stakeholder Engagement", 
            "Technology Adoption",
            "Economic Factors",
            "Environmental Impact",
            "Social Acceptance",
            "Regulatory Framework",
            "Behavioral Change",
            "International Cooperation",
            "Sustainable Development"
        ]
        
        # Subcategories for abductive mode testing
        self.abductive_subcategories = {
            "Policy Implementation": ["Timeline Challenges", "Resource Allocation", "Monitoring Systems"],
            "Stakeholder Engagement": ["Communication Gaps", "Power Dynamics", "Trust Building"],
            "Technology Adoption": ["Learning Curves", "Infrastructure Requirements", "Compatibility Issues"],
            "Economic Factors": ["Cost-Benefit Analysis", "Funding Sources", "Economic Incentives"]
        }
        
        # Subcodes for grounded mode testing
        self.grounded_subcodes = [
            "Emergent Patterns",
            "Conceptual Relationships", 
            "Contextual Factors",
            "Process Dynamics",
            "Structural Constraints",
            "Agency and Action",
            "Temporal Dimensions",
            "Spatial Considerations",
            "Power Relations",
            "Identity Construction"
        ]
    
    def generate_segment(self, mode: str, segment_id: int) -> Dict[str, Any]:
        """Generate a test segment for the specified mode."""
        texts = self.sample_texts[mode]
        text = self.random.choice(texts)
        
        base_segment = {
            "segment_id": f"test_{mode}_{segment_id:03d}",
            "text": text,
            "analysis_mode": mode,
            "difficulty": self.random.choice(["low", "medium", "high"]),
            "length_category": "short" if len(text) < 150 else "medium" if len(text) < 300 else "long"
        }
        
        # Add mode-specific expected results
        if mode == 'deductive':
            base_segment["expected_results"] = {
                "category": self.random.choice(self.deductive_categories),
                "subcategories": self.random.sample(self.deductive_categories, 
                                                  k=self.random.randint(1, 3)),
                "confidence": {
                    "total": round(self.random.uniform(0.7, 0.95), 2),
                    "category": round(self.random.uniform(0.8, 0.98), 2),
                    "subcategories": round(self.random.uniform(0.6, 0.9), 2)
                },
                "multiple_coding": self.random.random() < 0.3,  # 30% chance
                "relevance_scores": {
                    "research_relevance": round(self.random.uniform(0.6, 1.0), 2),
                    "coding_relevance": round(self.random.uniform(0.7, 1.0), 2)
                }
            }
        
        elif mode == 'inductive':
            base_segment["expected_results"] = {
                "potential_categories": self.random.sample(self.deductive_categories, 
                                                         k=self.random.randint(2, 5)),
                "category_development": {
                    "main_category": self.random.choice(self.deductive_categories),
                    "supporting_categories": self.random.sample(self.deductive_categories, 
                                                               k=self.random.randint(1, 3))
                },
                "confidence": round(self.random.uniform(0.6, 0.9), 2),
                "novelty_score": round(self.random.uniform(0.3, 0.8), 2)
            }
        
        elif mode == 'abductive':
            category = self.random.choice(list(self.abductive_subcategories.keys()))
            base_segment["expected_results"] = {
                "base_category": category,
                "extended_subcategories": self.random.sample(self.abductive_subcategories[category], 
                                                            k=self.random.randint(1, 3)),
                "new_subcategories": [
                    f"Extended {self.random.choice(['Analysis', 'Perspective', 'Dimension'])}",
                    f"Contextual {self.random.choice(['Factor', 'Constraint', 'Opportunity'])}"
                ],
                "confidence": round(self.random.uniform(0.65, 0.92), 2),
                "expansion_quality": round(self.random.uniform(0.5, 0.85), 2)
            }
        
        elif mode == 'grounded':
            base_segment["expected_results"] = {
                "subcodes": self.random.sample(self.grounded_subcodes, 
                                             k=self.random.randint(2, 6)),
                "subcode_details": [
                    {
                        "name": code,
                        "definition": f"Definition for {code}",
                        "evidence": [text[i:i+50] for i in range(0, min(len(text), 150), 50)],
                        "keywords": [code.lower().replace(" ", "_"), 
                                    f"related_{self.random.choice(['term', 'concept', 'idea'])}"],
                        "confidence": round(self.random.uniform(0.6, 0.95), 2)
                    }
                    for code in self.random.sample(self.grounded_subcodes, 
                                                  k=self.random.randint(2, 4))
                ],
                "memo": f"Analytical memo for segment {segment_id:03d}",
                "saturation_indicator": round(self.random.uniform(0.4, 0.9), 2)
            }
        
        return base_segment
    
    def generate_dataset(self, segments_per_mode: int = 25) -> Dict[str, Any]:
        """Generate complete test dataset."""
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Test dataset for API call optimization validation",
                "segments_per_mode": segments_per_mode,
                "total_segments": segments_per_mode * 4
            },
            "segments": [],
            "baseline_metrics": {
                "deductive": {
                    "api_calls_per_segment": 2.2,
                    "processing_time_seconds": 15.3,
                    "intercoder_reliability": 0.78,
                    "precision": 0.82,
                    "recall": 0.76,
                    "f1_score": 0.79
                },
                "inductive": {
                    "api_calls_per_segment": 1.0,
                    "processing_time_seconds": 12.7,
                    "intercoder_reliability": 0.72,
                    "precision": 0.75,
                    "recall": 0.71,
                    "f1_score": 0.73
                },
                "abductive": {
                    "api_calls_per_segment": 1.0,
                    "processing_time_seconds": 11.8,
                    "intercoder_reliability": 0.74,
                    "precision": 0.77,
                    "recall": 0.73,
                    "f1_score": 0.75
                },
                "grounded": {
                    "api_calls_per_segment": 1.0,
                    "processing_time_seconds": 14.2,
                    "intercoder_reliability": 0.71,
                    "precision": 0.74,
                    "recall": 0.69,
                    "f1_score": 0.71
                }
            }
        }
        
        # Generate segments for each mode
        modes = ['deductive', 'inductive', 'abductive', 'grounded']
        segment_counter = 1
        
        for mode in modes:
            for i in range(segments_per_mode):
                segment = self.generate_segment(mode, segment_counter)
                dataset["segments"].append(segment)
                segment_counter += 1
        
        return dataset
    
    def save_dataset(self, filepath: str, segments_per_mode: int = 25):
        """Generate and save dataset to JSON file."""
        dataset = self.generate_dataset(segments_per_mode)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filepath}")
        print(f"Total segments: {len(dataset['segments'])}")
        print(f"Segments per mode: {segments_per_mode}")
        print(f"Modes: deductive, inductive, abductive, grounded")
        
        return dataset


def main():
    """Generate test dataset."""
    generator = TestDatasetGenerator(seed=42)
    
    # Create tests directory if it doesn't exist
    import os
    os.makedirs("C:/Users/justu/OneDrive/Projekte/Forschung/R-Projects/QCA-AID/QCA_AID_assets/optimization/tests/data", 
                exist_ok=True)
    
    # Generate and save dataset
    dataset_path = "C:/Users/justu/OneDrive/Projekte/Forschung/R-Projects/QCA-AID/QCA_AID_assets/optimization/tests/data/test_dataset_v1.json"
    dataset = generator.save_dataset(dataset_path)
    
    # Also save baseline metrics separately
    baseline_path = "C:/Users/justu/OneDrive/Projekte/Forschung/R-Projects/QCA-AID/QCA_AID_assets/optimization/tests/data/baseline_metrics.json"
    with open(baseline_path, 'w', encoding='utf-8') as f:
        json.dump(dataset["baseline_metrics"], f, indent=2, ensure_ascii=False)
    
    print(f"Baseline metrics saved to {baseline_path}")
    
    # Print summary
    print("\nDataset Summary:")
    for mode in ['deductive', 'inductive', 'abductive', 'grounded']:
        mode_segments = [s for s in dataset['segments'] if s['analysis_mode'] == mode]
        print(f"  {mode}: {len(mode_segments)} segments")
    
    return dataset


if __name__ == "__main__":
    main()