"""
Synthetic Insurance Policy Fragment Dataset Generator

This script generates 5000 synthetic insurance policy text fragments with labeled 
suspicious content for training fraud detection models. It creates policy clauses 
and fragments with various types of red flags at document, section, and clause levels.

Features:
- Generates 5000 synthetic insurance policy fragments
- Multi-level labeling (document, section, clause)
- Realistic policy fragment structure
- Configurable red flag injection ratio
- Text file output as list of dictionaries
- No real personal data - all synthetic

Author: AI Assistant
Date: 2025-09-13
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class InsurancePolicyFragmentGenerator:
    """
    Generates synthetic insurance policy fragments with realistic content and suspicious clauses.
    """
    
    def __init__(self, num_fragments: int = 5000, red_flag_ratio: float = 0.3):
        """
        Initialize the policy fragment generator.
        
        Args:
            num_fragments: Number of policy fragments to generate
            red_flag_ratio: Proportion of fragments that should contain red flags (0.0 to 1.0)
        """
        self.num_fragments = num_fragments
        self.red_flag_ratio = red_flag_ratio
        
        # Initialize data pools for realistic content generation
        self._init_data_pools()
        
        # Define red flag patterns and normal patterns
        self._init_fragment_patterns()
    
    def _init_data_pools(self):
        """Initialize pools of realistic data for generating policy content."""
        
        # Insurance companies (all synthetic)
        self.insurance_companies = [
            "SecureLife Insurance Co.", "TrustGuard Insurance", "SafeHaven Mutual",
            "ProTech Insurance Group", "Guardian Shield Insurance", "Reliable Coverage Corp.",
            "Premier Protection Inc.", "Unity Insurance Services", "Cornerstone Insurance",
            "Beacon Insurance Company", "Fortress Insurance Group", "Pinnacle Coverage LLC",
            "Apex Insurance Solutions", "Sterling Protection Services", "Liberty Shield",
            "Crown Insurance Partners", "Diamond Coverage Group", "Eagle Eye Insurance"
        ]
        
        # Coverage types and amounts for context
        self.coverage_types = [
            "Life Insurance", "Auto Insurance", "Home Insurance", "Health Insurance",
            "Disability Insurance", "Travel Insurance", "Business Insurance", "Renters Insurance",
            "Umbrella Insurance", "Professional Liability Insurance"
        ]
        
        # Common insurance terms and amounts
        self.coverage_amounts = ["$25,000", "$50,000", "$100,000", "$250,000", "$500,000", "$1,000,000"]
        self.time_periods = ["30 days", "60 days", "90 days", "6 months", "1 year", "2 years"]
        self.percentages = ["10%", "15%", "20%", "25%", "50%", "75%"]
        
    def _init_fragment_patterns(self):
        """Initialize patterns for suspicious and normal policy fragments."""
        
        # Step 2: Define suspicious clause patterns (red flags)
        self.suspicious_fragments = {
            "coverage": [
                "Coverage may be denied at the sole discretion of the company without explanation or right to appeal.",
                "Benefits are subject to undisclosed internal review processes that may take indefinite time to complete.",
                "Coverage limits may be adjusted retroactively based on proprietary company risk assessment algorithms.",
                "The company reserves the right to redefine covered events without prior written notice to policyholders.",
                "Pre-existing conditions include any medical condition not explicitly pre-approved in writing by our medical review board.",
                "Coverage applies only during specific time periods that will be disclosed at the company's sole discretion.",
                "Benefits may be reduced or eliminated based on social media activity and lifestyle choices as determined by our AI monitoring system.",
                "Coverage becomes void if the policyholder fails to maintain a minimum credit score of 750 at all times.",
                "All coverage decisions are final and not subject to external review, arbitration, or legal challenge.",
                "Coverage may be suspended without notice if the policyholder's occupation changes in any way."
            ],
            
            "exclusions": [
                "This policy excludes coverage for any incident occurring on days ending in 'y' or during any leap year.",
                "Exclusions may be added retroactively at any time without policyholder notification, consent, or compensation.",
                "Coverage is permanently void if the policyholder has ever filed any insurance claim with any company worldwide.",
                "This policy does not cover incidents that could theoretically have been prevented by any unspecified means.",
                "Exclusions include any event not specifically anticipated and documented in writing by the policyholder.",
                "Coverage is excluded for incidents occurring in any location not pre-approved in writing by our risk assessment team.",
                "Any action taken during periods of emotional distress, as determined by the company, automatically voids all coverage.",
                "Claims are excluded if the policyholder cannot provide continuous video surveillance footage of the entire incident.",
                "Coverage does not apply if the incident occurs while the policyholder is wearing certain colors as determined by company policy.",
                "Exclusions apply to any incident that occurs during months containing the letter 'r'."
            ],
            
            "premium": [
                "Premium rates may increase by up to 500% annually without advance notification, justification, or policyholder consent.",
                "A non-refundable processing fee of $5,000 is required before any claim review can commence, regardless of claim validity.",
                "Premiums must be paid 90 days in advance or coverage is immediately and permanently terminated without refund.",
                "Late payment fees compound daily at 25% interest with no maximum cap or payment plan options.",
                "Premium calculations are based on confidential proprietary factors that are never disclosed to policyholders or regulators.",
                "A mandatory 'administrative convenience fee' of $200 is charged monthly regardless of policy activity or claims filed.",
                "Premiums are subject to immediate adjustment based on undisclosed market conditions and internal company financial needs.",
                "Failure to pay premiums within 24 hours of the due date results in permanent policy cancellation and forfeiture of all paid premiums.",
                "Premium refunds are never provided under any circumstances, including policy cancellation, death, or company error.",
                "Additional surcharges may be applied at any time based on the policyholder's zip code, age, or marital status changes."
            ],
            
            "claims": [
                "All claims must be filed within 24 hours of the incident or they are automatically and permanently denied without review.",
                "Claims require notarized statements from at least 5 independent witnesses who are not related to or acquainted with the policyholder.",
                "The claims review process may take up to 10 years for completion with no interim payments or status updates provided.",
                "Policyholders must provide DNA evidence, polygraph test results, and medical examinations for all claims regardless of type or amount.",
                "Claims are processed only on the second Tuesday of months with 31 days and only during full moon phases.",
                "All claims must be submitted in person at our single office location in Antarctica with no exceptions or alternatives.",
                "Claims require a $10,000 non-refundable investigation fee paid in cash before any review process can begin.",
                "Policyholders must appear before our claims tribunal wearing formal business attire and recite our 50-page company pledge from memory.",
                "Claims are automatically denied if submitted using black ink instead of blue ink or if any form contains correction fluid.",
                "All claim documentation must be handwritten in cursive using a fountain pen and submitted on parchment paper."
            ],
            
            "misc": [
                "This policy becomes permanently void if the policyholder changes their hairstyle, hair color, or facial hair without written company approval.",
                "The company may cancel this policy at any time if the policyholder's astrological chart is deemed unfavorable by our consulting astrologer.",
                "Policy terms are subject to change based on the company CEO's daily mood, personal preferences, and horoscope readings.",
                "Policyholders must maintain an active social media presence with at least 1000 followers and post daily updates or face immediate cancellation.",
                "This agreement is governed by the laws of a jurisdiction to be determined at the company's sole discretion after any dispute arises.",
                "The company reserves the right to require annual polygraph tests, home inspections, and psychological evaluations at the policyholder's expense.",
                "Policy interpretation is based exclusively on the company's proprietary dictionary and legal definitions that are not available to policyholders.",
                "Policyholders must agree to participate in company-sponsored medical experiments, drug trials, and research studies upon written request.",
                "This policy is automatically void if the policyholder ever speaks negatively about the company in any public or private context.",
                "Coverage requires the policyholder to maintain a specific body weight, exercise routine, and dietary restrictions as determined by company wellness standards."
            ]
        }
        
        # Step 2: Define normal, legitimate policy fragments for comparison
        self.normal_fragments = {
            "coverage": [
                "This policy provides comprehensive coverage for the specified risks outlined in the policy schedule and declarations page.",
                "Coverage begins on the effective date and continues while premiums are paid according to the agreed payment schedule.",
                "Benefits are paid according to the benefit schedule outlined in the policy documents within 30 business days of claim approval.",
                "Coverage includes protection against the perils specifically listed and clearly defined in this policy document.",
                "This insurance provides financial protection up to the policy limits specified in the declarations page of your policy.",
                "Coverage applies 24 hours a day, 365 days a year, subject to the terms and conditions clearly outlined in this policy.",
                "Benefits are calculated based on the coverage amounts and deductibles selected by the policyholder at the time of application.",
                "This policy covers both sudden and accidental losses as specifically defined in the policy terms and conditions section.",
                "Coverage territory includes the United States and its territories unless otherwise specified in the policy documents.",
                "Benefits under this policy are subject to the maximum limits shown in the policy schedule and are not cumulative across policy periods."
            ],
            
            "exclusions": [
                "This policy does not cover losses due to war, nuclear hazards, or government-declared acts of terrorism as defined by federal law.",
                "Intentional acts by the policyholder or beneficiaries are excluded from coverage under this insurance policy.",
                "Losses occurring outside the policy territory as clearly defined in the policy documents are not covered.",
                "Pre-existing medical conditions diagnosed before the effective date of coverage are excluded from health insurance benefits.",
                "Normal wear and tear, gradual deterioration, and routine maintenance issues are not covered under this policy.",
                "Losses due to illegal activities or criminal acts by the policyholder are excluded from coverage.",
                "Damage caused by floods, earthquakes, or other natural disasters may require separate coverage as specified in your policy.",
                "Business-related losses are excluded unless specifically covered under a separate business insurance policy.",
                "Losses resulting from the policyholder's professional services are excluded unless professional liability coverage is included.",
                "Damage to property caused by pets or domestic animals is excluded unless specifically covered by an endorsement."
            ],
            
            "premium": [
                "Premiums are due on the dates specified in the policy schedule and will be clearly indicated on all billing statements.",
                "A 30-day grace period is provided for premium payments, during which coverage remains in full effect.",
                "Premium rates are guaranteed for the initial policy term as specified in your policy documents and declarations page.",
                "Discounts may be available for multiple policies, safe driving records, security systems, or other qualifying factors.",
                "Premium refunds are calculated on a pro-rata basis if the policy is cancelled before the expiration date.",
                "Premium adjustments may occur at renewal based on claims experience, current rates, and changes in coverage.",
                "Payment options include monthly, quarterly, semi-annual, or annual payment plans for your convenience.",
                "Automatic payment options are available through electronic funds transfer with advance notice of any rate changes.",
                "Premium notices will be sent at least 30 days before the due date to ensure adequate time for payment.",
                "Any premium increases will be communicated in writing at least 45 days before the renewal date."
            ],
            
            "claims": [
                "Claims should be reported as soon as reasonably possible, but no later than 30 days after the loss occurs.",
                "The company will investigate all claims promptly, fairly, and in accordance with applicable state and federal laws.",
                "Claims are typically processed within 30 business days of receiving all required documentation and information.",
                "Policyholders have the right to appeal claim decisions through our formal review process outlined in the policy.",
                "All necessary claim forms and instructions are available on our website or by calling our customer service department.",
                "The company may require reasonable documentation, inspections, or examinations as part of the standard claims process.",
                "Settlement options include repair, replacement, or cash payment based on policy terms and policyholder preference.",
                "Claims representatives are available to assist policyholders throughout the entire claims process.",
                "Independent adjusters may be assigned to evaluate claims and ensure fair and accurate settlements.",
                "Claim payments will be made promptly upon completion of the investigation and determination of coverage."
            ],
            
            "misc": [
                "This policy is governed by the laws of the state where it was issued and where the policyholder maintains residence.",
                "Any disputes will be resolved through binding arbitration or court proceedings as specified by applicable state law.",
                "Policy changes require written agreement from both the insurance company and the policyholder.",
                "This policy constitutes the entire agreement between the parties and supersedes all prior agreements and understandings.",
                "The company is licensed and regulated by the state insurance department and subject to regular regulatory oversight.",
                "Policyholders have certain rights under state insurance laws, including the right to file complaints with state regulators.",
                "This policy may be cancelled by either party with appropriate advance notice as required by state insurance law.",
                "The company maintains adequate financial reserves and reinsurance to ensure the ability to pay all valid claims.",
                "Policy documents are available in multiple languages upon request to ensure understanding of coverage terms.",
                "Customer service representatives are available during normal business hours to answer questions about your policy."
            ]
        }
        
        # Step 2: Additional fragment types for variety
        self.fragment_types = ["coverage", "exclusions", "premium", "claims", "misc"]
    
    def _generate_fragment(self, fragment_id: int) -> Dict[str, Any]:
        """
        Generate a single insurance policy fragment.
        
        Args:
            fragment_id: Unique identifier for the fragment
            
        Returns:
            Dictionary containing fragment text and label
        """
        
        # Step 3: Determine if this fragment should have red flags
        has_red_flags = random.random() < self.red_flag_ratio
        
        # Step 3: Select fragment type and content
        fragment_type = random.choice(self.fragment_types)
        
        if has_red_flags:
            # Step 3: Choose a suspicious fragment
            fragment_text = random.choice(self.suspicious_fragments[fragment_type])
        else:
            # Step 3: Choose a normal fragment
            fragment_text = random.choice(self.normal_fragments[fragment_type])
        
        # Step 3: Add some context to make it more realistic
        company = random.choice(self.insurance_companies)
        coverage_type = random.choice(self.coverage_types)
        
        # Step 3: Create contextual fragment text
        context_intro = random.choice([
            f"Under this {coverage_type} policy issued by {company}: ",
            f"According to the terms of your {coverage_type} coverage: ",
            f"This {coverage_type} policy states that: ",
            f"As outlined in your {company} policy: ",
            f"The following clause applies to your {coverage_type}: ",
            ""  # Sometimes no intro for variety
        ])
        
        full_text = context_intro + fragment_text
        
        # Step 4: Return simplified format with only 'text' and 'label' keys
        return {
            "text": full_text,
            "label": 1 if has_red_flags else 0
        }
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """
        Generate the complete dataset of insurance policy fragments.
        
        Returns:
            List of fragment dictionaries with 'text' and 'label' keys
        """
        
        print(f"🏥 Generating {self.num_fragments:,} synthetic insurance policy fragments...")
        print(f"🎯 Target red flag ratio: {self.red_flag_ratio:.1%}")
        print("=" * 70)
        
        fragments = []
        
        # Step 1: Generate the specified number of fragments
        for i in range(self.num_fragments):
            if (i + 1) % 500 == 0:
                print(f"📝 Generated {i + 1:,}/{self.num_fragments:,} fragments...")
            
            fragment = self._generate_fragment(i + 1)
            fragments.append(fragment)
        
        # Step 6: Calculate actual statistics
        total_fragments = len(fragments)
        suspicious_fragments = sum(1 for f in fragments if f["label"] == 1)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"📊 Final Statistics:")
        print(f"   📄 Total fragments: {total_fragments:,}")
        print(f"   🚩 Suspicious fragments: {suspicious_fragments:,} ({suspicious_fragments/total_fragments:.1%})")
        print(f"   ✅ Normal fragments: {total_fragments - suspicious_fragments:,} ({(total_fragments - suspicious_fragments)/total_fragments:.1%})")
        
        return fragments
    
    def save_dataset(self, fragments: List[Dict[str, Any]], filename: str = "insurance_fragments.txt"):
        """
        Save the generated dataset to a text file as a list of dictionaries.
        
        Args:
            fragments: List of fragment dictionaries with 'text' and 'label' keys
            filename: Output filename
        """
        
        print(f"\n💾 Saving dataset to {filename}...")
        
        # Step 4: Ensure the file path is in the classifier directory
        if not filename.startswith('/'):
            filename = os.path.join(os.path.dirname(__file__), filename)
        
        # Step 4: Save as text file with list of dictionaries
        with open(filename, 'w', encoding='utf-8') as f:
            # Write as a properly formatted list of dictionaries
            f.write("[\n")
            for i, fragment in enumerate(fragments):
                # Write each dictionary on its own line with proper JSON formatting
                json_line = json.dumps(fragment, ensure_ascii=False)
                if i < len(fragments) - 1:
                    f.write(f"  {json_line},\n")
                else:
                    f.write(f"  {json_line}\n")
            f.write("]\n")
        
        print(f"✅ Dataset saved successfully!")
        print(f"📊 Format: Text file containing list of {len(fragments):,} dictionaries")
        print(f"📊 Each dictionary has 'text' and 'label' keys")
        
        # Also save a sample for quick inspection
        sample_filename = filename.replace('.txt', '_sample.txt')
        sample_fragments = fragments[:10]  # First 10 fragments
        
        with open(sample_filename, 'w', encoding='utf-8') as f:
            f.write("[\n")
            for i, fragment in enumerate(sample_fragments):
                json_line = json.dumps(fragment, ensure_ascii=False)
                if i < len(sample_fragments) - 1:
                    f.write(f"  {json_line},\n")
                else:
                    f.write(f"  {json_line}\n")
            f.write("]\n")
        
        print(f"📄 Sample dataset (10 fragments) saved to {sample_filename}")
        
        # Print file sizes
        main_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        sample_size = os.path.getsize(sample_filename) / 1024  # KB
        print(f"📏 File sizes: Main dataset: {main_size:.1f} MB, Sample: {sample_size:.1f} KB")


def main():
    """
    Main function to generate and save the synthetic insurance policy fragment dataset.
    
    Step 6: Configuration options - easily scalable:
    - NUM_FRAGMENTS: Number of fragments to generate (default: 5000)
    - RED_FLAG_RATIO: Proportion of fragments with suspicious content (default: 0.3)
    - OUTPUT_FILENAME: Name of the output text file
    """
    
    # Step 6: Configuration - easily adjustable parameters for scaling
    NUM_FRAGMENTS = 5000  # Number of policy fragments to generate
    RED_FLAG_RATIO = 0.3   # 30% of fragments will contain suspicious content
    OUTPUT_FILENAME = "insurance_fragments.txt"
    
    print("🏥 Synthetic Insurance Policy Fragment Dataset Generator")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Fragments to generate: {NUM_FRAGMENTS:,}")
    print(f"   Red flag ratio: {RED_FLAG_RATIO:.1%}")
    print(f"   Output file: {OUTPUT_FILENAME}")
    print("=" * 70)
    
    # Step 1: Initialize generator with configurable parameters
    generator = InsurancePolicyFragmentGenerator(
        num_fragments=NUM_FRAGMENTS,
        red_flag_ratio=RED_FLAG_RATIO
    )
    
    # Step 1: Generate dataset
    fragments = generator.generate_dataset()
    
    # Step 4: Save to text file
    generator.save_dataset(fragments, OUTPUT_FILENAME)
    
    print(f"\n🎉 All done! Your synthetic insurance policy fragment dataset is ready.")
    print(f"📁 Files created:")
    print(f"   - {OUTPUT_FILENAME} (complete dataset)")
    print(f"   - {OUTPUT_FILENAME.replace('.txt', '_sample.txt')} (sample for inspection)")
    
    # Display a sample fragment for verification
    print(f"\n📋 Sample Fragment Preview:")
    print("=" * 70)
    sample_fragment = fragments[0]
    print(f"Label: {sample_fragment['label']} ({'SUSPICIOUS' if sample_fragment['label'] == 1 else 'NORMAL'})")
    
    print(f"\n📄 Fragment Text:")
    print("-" * 70)
    print(f"'{sample_fragment['text']}'")
    
    # Show a suspicious example if available
    suspicious_examples = [f for f in fragments[:50] if f['label'] == 1]
    if suspicious_examples:
        print(f"\n🚩 Example Suspicious Fragment:")
        print("-" * 70)
        print(f"'{suspicious_examples[0]['text']}'")
    
    print(f"\n💡 Usage Tips:")
    print(f"   - Load the text file directly in your ML training pipeline")
    print(f"   - Each dictionary has 'text' (fragment content) and 'label' (0=normal, 1=suspicious)")
    print(f"   - Use: import ast; data = ast.literal_eval(open('insurance_fragments.txt').read())")
    print(f"   - Or: import json; data = json.load(open('insurance_fragments.txt'))")
    print(f"   - Split data: train_data = data[:4000], val_data = data[4000:]")
    print(f"   - Adjust NUM_FRAGMENTS and RED_FLAG_RATIO as needed for scaling")


if __name__ == "__main__":
    main()
