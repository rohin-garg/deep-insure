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
  "**Q:** What's my annual deductible?\n**A:** Your annual deductible is $1,500 for individual coverage and $3,000 for family coverage. [See deductibles section](deductibles-copays)",
  "**Q:** Are emergency room visits covered?\n**A:** Yes, [emergency services](coverage-overview) are covered 24/7 with a $200 copay after you meet your deductible.",
  "**Q:** How much do prescription drugs cost?\n**A:** Prescription costs vary by tier: Generic drugs are $10, preferred brands $35, non-preferred brands $70, and specialty drugs are 25% coinsurance. [See prescription coverage](prescription-coverage)"
];