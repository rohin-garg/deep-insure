export interface InsuranceSection {
  id: string;
  header: string;
  text: string;
}

export const mockSummary: InsuranceSection[] = [
  {
    id: "coverage-overview",
    header: "Coverage Overview",
    text: `# Coverage Overview

Your plan includes comprehensive medical coverage with the following key benefits:

## Medical Services
- **Preventive Care**: 100% covered when using in-network providers
- **Primary Care Visits**: $25 copay per visit
- **Specialist Visits**: $50 copay per visit
- **Urgent Care**: $75 copay per visit
- **Emergency Room**: $300 copay (waived if admitted)

## Hospital Services
- **Inpatient Hospital**: 80% covered after deductible
- **Outpatient Surgery**: 80% covered after deductible
- **Diagnostic Tests**: 80% covered after deductible

## Additional Benefits
- **Mental Health Services**: Same as medical benefits
- **Maternity Care**: Covered at 100% after deductible
- **Physical Therapy**: $40 copay per visit (up to 20 visits per year)

## Extended Coverage Details

### Surgical Procedures
Your insurance covers a wide range of surgical procedures both inpatient and outpatient. This includes:
- **Cardiac Surgery**: Full coverage after meeting your annual deductible
- **Orthopedic Surgery**: Including joint replacements, arthroscopic procedures
- **General Surgery**: Appendectomy, hernia repair, gallbladder removal
- **Neurosurgery**: Brain and spinal cord procedures with prior authorization
- **Plastic Surgery**: Reconstructive procedures covered, cosmetic procedures excluded

### Rehabilitation Services
- **Occupational Therapy**: $40 copay per visit (up to 30 visits per year)
- **Speech Therapy**: $40 copay per visit (up to 20 visits per year)
- **Cardiac Rehabilitation**: 80% covered after deductible
- **Pulmonary Rehabilitation**: 80% covered after deductible

### Durable Medical Equipment
- **Wheelchairs and Mobility Aids**: 80% covered after deductible
- **Oxygen Equipment**: 80% covered after deductible
- **Prosthetic Devices**: 80% covered after deductible
- **Diabetic Supplies**: Covered at 100% when obtained through preferred vendors

### Vision and Hearing
- **Routine Eye Exams**: One per year at $10 copay
- **Eyeglasses/Contact Lenses**: $150 allowance every two years
- **Hearing Aids**: 80% covered up to $2,500 per ear every three years
- **Hearing Exams**: Covered same as specialist visits

### Alternative Medicine
- **Chiropractic Care**: $30 copay per visit (up to 12 visits per year)
- **Acupuncture**: $40 copay per visit (up to 10 visits per year)
- **Massage Therapy**: Not covered unless medically necessary

### Home Health Care
- **Skilled Nursing**: 80% covered after deductible (up to 100 visits per year)
- **Home Health Aide**: 80% covered after deductible (up to 40 hours per week)
- **Medical Equipment**: Same as durable medical equipment coverage

### Hospice and Palliative Care
- **Hospice Services**: 100% covered when certified as terminally ill
- **Palliative Care**: Covered same as medical benefits
- **Respite Care**: Limited coverage for family caregiver relief

*Note: All percentages and copays are based on in-network providers. Out-of-network benefits may be reduced. Prior authorization may be required for certain services.*`
  },
  {
    id: "deductibles-copays",
    header: "Deductibles & Copays", 
    text: `# Deductibles & Copays

Understanding your cost-sharing responsibilities helps you budget for healthcare expenses throughout the year.

## Annual Deductible
- **Individual**: $1,500 per year
- **Family**: $3,000 per year

You must meet your deductible before insurance begins paying for most services (except preventive care).

## Copayments
- **Primary Care Visit**: $25
- **Specialist Visit**: $50
- **Emergency Room**: $200
- **Urgent Care**: $75

[Prescription drugs](prescription-link) have varying copays based on the medication tier. Generic drugs typically have the lowest copays.

## Out-of-Pocket Maximum
- **Individual**: $6,000
- **Family**: $12,000

Once you reach this limit, insurance pays 100% of covered services for the rest of the year.`
  },
  {
    id: "network-providers",
    header: "Network Providers",
    text: `# Network Providers

Using in-network providers ensures you receive the maximum benefits and lowest costs under your insurance plan.

## In-Network Benefits
- **Lower costs**: Reduced deductibles and copayments
- **Pre-negotiated rates**: Providers accept agreed-upon fees
- **Streamlined billing**: Direct billing between provider and insurance

## Finding Providers
[Find a provider](provider-directory-link) using our online directory:
- Search by specialty, location, or name
- Verify current network status
- Read patient reviews and ratings
- Check appointment availability

## Out-of-Network Coverage
While we provide some coverage for out-of-network care, your costs will be significantly higher:
- Higher deductibles and coinsurance
- Balance billing may apply
- Prior authorization often required

## Referrals
Some services may require referrals from your [primary care physician](pcp-link) to ensure coverage.`
  },
  {
    id: "prescription-coverage",
    header: "Prescription Coverage",
    text: `# Prescription Drug Coverage

Your prescription drug benefit provides access to a comprehensive formulary of medications at affordable costs.

## Formulary Tiers
- **Tier 1 (Generic)**: $10 copay
- **Tier 2 (Preferred Brand)**: $35 copay  
- **Tier 3 (Non-Preferred Brand)**: $70 copay
- **Tier 4 (Specialty)**: 25% coinsurance

## Pharmacy Network
- **Retail pharmacies**: 30-day supply
- **Mail-order pharmacy**: 90-day supply at reduced cost
- **Specialty pharmacy**: For complex medications

[Prior authorization](prior-auth-link) may be required for certain high-cost medications. Your doctor can help with the approval process.

## Cost-Saving Tips
- Choose generic medications when available
- Use mail-order for maintenance medications
- Ask about therapeutic alternatives`
  }
];

export const mockChatHistory = [
  "**User:** What's my annual deductible?",
  "**Assistant:** Your annual deductible is $1,500 for individual coverage and $3,000 for family coverage. This means you'll pay the full cost of covered services until you reach this amount each year. After that, your insurance will start sharing costs with you through copayments and coinsurance.",
  "**User:** Are emergency room visits covered?",
  "**Assistant:** Yes, emergency room visits are covered 24/7 with a $300 copay (waived if you're admitted to the hospital). However, if your condition isn't truly an emergency, consider using urgent care first, which only has a $75 copay and can handle many medical situations.",
  "**User:** How much do prescription drugs cost?",
  "**Assistant:** Prescription costs vary by tier: Generic drugs (Tier 1) have a $10 copay, preferred brand drugs (Tier 2) are $35, non-preferred brands (Tier 3) cost $70, and specialty drugs (Tier 4) require 25% coinsurance. You can save money by choosing generics when available or using mail-order pharmacy for 90-day supplies."
];

// Mock AI responses for different scenarios
export const mockAIResponse = `Based on your insurance policy documents, I can provide you with comprehensive information about your coverage.

Your policy includes several key benefits:

**Coverage Highlights:**
- Preventive care is covered at 100% when using in-network providers [1]
- Primary care visits have a $25 copay [2]
- Specialist visits require a $50 copay [2]
- Emergency room visits have a $300 copay (waived if admitted) [1]

**Detailed Benefits:**
Your plan covers a wide range of medical services including hospital stays, outpatient surgery, diagnostic tests, and prescription medications [3]. Mental health services are covered at the same level as medical benefits [1].

For more specific information about deductibles and copays [2], please refer to your policy documents. You can also review information about network providers [3] to ensure you're maximizing your benefits.

**Important Notes:**
- All percentages and copays listed are for in-network providers [3]
- Out-of-network services may have reduced coverage [1]
- Prior authorization may be required for certain procedures [2]

Would you like me to elaborate on any specific aspect of your coverage?`;

export const mockFollowUpResponses = [
  `Thank you for your follow-up question. Let me provide more specific details about that aspect of your coverage.

**Additional Coverage Details:**
- Maternity and newborn care is covered at 100% after you meet your deductible [1]
- Mental health and substance abuse services are covered at the same level as medical benefits [2]
- Physical therapy requires a $40 copay per visit with up to 20 visits covered per year [3]

**Cost-Saving Tips:**
- Always verify that your provider is in-network before receiving services [3]
- Consider using urgent care instead of the emergency room for non-emergency situations [1]
- Take advantage of free preventive services like annual check-ups and screenings [2]

Is there anything else about your benefits you'd like me to clarify?`,

  `I'm glad you're asking for more information. Here are additional details that might be helpful:

**Prescription Drug Benefits:**
- Generic medications have the lowest copays at just $10 per prescription [1]
- Mail-order pharmacy offers 90-day supplies at reduced costs [2]
- Prior authorization may be required for certain high-cost medications [3]

**Wellness Programs:**
- Free annual wellness exam with no copay [1]
- Discounts available for gym memberships and wellness activities [2]
- Tobacco cessation programs covered at 100% [3]

**Special Services:**
- Telemedicine visits available 24/7 with reduced copays [2]
- Second opinion consultations covered for major diagnoses [1]
- Case management services for complex medical conditions [3]

Would you like more details about any of these specific benefits?`
];

export const mockSourceCards = [
  {
    title: "UnitedHealthcare Policy Document",
    url: "policy-doc-1.pdf",
    snippet: `Section 4.2: Coverage Overview

Your plan includes comprehensive medical coverage with preventive care at 100% when using in-network providers. Primary care visits require a $25 copay per visit, while specialist consultations have a $50 copay.

Emergency services are available 24/7 with a $300 copay, which is waived if you are admitted to the hospital. All emergency care is covered regardless of network status, as required by federal law.

Mental health and substance abuse services are covered at the same benefit level as medical services, ensuring parity in your healthcare coverage.`,
    type: "PDF Document"
  },
  {
    title: "Benefits Summary",
    url: "benefits-summary.pdf",
    snippet: `Deductibles and Copays

Annual Deductible: $1,500 individual / $3,000 family
Out-of-pocket Maximum: $6,000 individual / $12,000 family

Copayments:
- Primary Care: $25
- Specialist: $50
- Urgent Care: $75
- Emergency Room: $300 (waived if admitted)

Coinsurance: 80% coverage after deductible for most services
Prior Authorization: Required for certain procedures and high-cost medications`,
    type: "Benefits Document"
  },
  {
    title: "Network Provider Directory",
    url: "provider-directory.html",
    snippet: `In-Network Providers

To maximize your benefits and minimize out-of-pocket costs, always use in-network healthcare providers. Your network includes over 1.2 million physicians and healthcare professionals nationwide.

Provider Search Features:
- Search by specialty, location, or provider name
- Real-time network status verification
- Patient reviews and quality ratings
- Appointment availability indicators

Out-of-Network Costs: Using providers outside your network will result in higher deductibles, increased coinsurance rates, and potential balance billing charges.`,
    type: "Provider Directory"
  },
  {
    title: "Prescription Drug Formulary",
    url: "drug-formulary.pdf",
    snippet: `Prescription Drug Coverage Tiers

Tier 1 Medications (Generic): $10 copay
- Most cost-effective option when available
- Bioequivalent to brand-name drugs
- Preferred by insurance for cost savings

Tier 2 Medications (Preferred Brand): $35 copay
- Brand-name drugs with negotiated rates
- Often have generic alternatives available

Tier 3 Medications (Non-Preferred Brand): $70 copay
- Higher-cost brand medications
- May require prior authorization

Tier 4 Medications (Specialty): 25% coinsurance
- Complex medications requiring special handling
- Often used for rare or chronic conditions
- May require specialty pharmacy dispensing`,
    type: "Formulary Document"
  },
  {
    title: "Wellness Benefits Guide",
    url: "wellness-benefits.pdf",
    snippet: `Preventive Care Services - 100% Covered

Annual Services (No Copay):
- Comprehensive physical examination
- Routine screenings (mammography, colonoscopy, etc.)
- Immunizations and vaccinations
- Well-woman visits including contraceptive counseling

Health Promotion Programs:
- Weight management counseling
- Tobacco cessation programs
- Diabetes prevention programs
- Stress management resources

Wellness Discounts:
- Gym membership reimbursement up to $200/year
- Fitness tracker subsidies
- Healthy lifestyle coaching programs`,
    type: "Wellness Guide"
  }
];

// Mock function to generate random chat ID
export const generateMockChatId = (): string => {
  return `mock-chat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};