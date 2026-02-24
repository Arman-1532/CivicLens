"""
Dataset merging module for complaint classification.
Handles loading and merging multiple datasets into a unified format.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

from .clean_data import clean_dataframe, remove_outliers
from .label_mapping import map_labels_in_dataframe, get_unified_categories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


# Dataset configurations - maps filename to column names
DATASET_CONFIGS: Dict[str, Dict[str, str]] = {
    "kaggle_customer_complaints.csv": {
        "text_column": "complaint",
        "label_column": "category"
    },
    "kaggle_public_service.csv": {
        "text_column": "description",
        "label_column": "type"
    },
    "kaggle_consumer_complaints.csv": {
        "text_column": "consumer_complaint_narrative",
        "label_column": "product"
    }
}


def load_dataset(
    filepath: Path,
    text_column: str,
    label_column: str,
    sample_size: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Load a single dataset and standardize column names.

    Args:
        filepath: Path to the CSV file
        text_column: Name of the text column in the source file
        label_column: Name of the label column in the source file
        sample_size: Optional sample size to limit data

    Returns:
        DataFrame with standardized columns or None if loading fails
    """
    if not filepath.exists():
        logger.warning(f"Dataset not found: {filepath}")
        return None

    try:
        logger.info(f"Loading dataset: {filepath.name}")
        df = pd.read_csv(filepath, low_memory=False)

        # Check required columns exist
        if text_column not in df.columns:
            logger.warning(f"Text column '{text_column}' not found in {filepath.name}")
            # Try to find similar column
            for col in df.columns:
                if 'text' in col.lower() or 'complaint' in col.lower() or 'description' in col.lower():
                    text_column = col
                    logger.info(f"Using alternative text column: {text_column}")
                    break
            else:
                return None

        if label_column not in df.columns:
            logger.warning(f"Label column '{label_column}' not found in {filepath.name}")
            # Try to find similar column
            for col in df.columns:
                if 'category' in col.lower() or 'type' in col.lower() or 'label' in col.lower() or 'product' in col.lower():
                    label_column = col
                    logger.info(f"Using alternative label column: {label_column}")
                    break
            else:
                return None

        # Standardize column names
        df_standardized = pd.DataFrame({
            'text': df[text_column],
            'category': df[label_column],
            'source': filepath.name
        })

        # Sample if requested
        if sample_size and len(df_standardized) > sample_size:
            df_standardized = df_standardized.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows from {filepath.name}")

        logger.info(f"Loaded {len(df_standardized)} rows from {filepath.name}")
        return df_standardized

    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def merge_datasets(
    datasets: List[pd.DataFrame],
    balance_classes: bool = True,
    min_samples_per_class: int = 100
) -> pd.DataFrame:
    """
    Merge multiple datasets into one.

    Args:
        datasets: List of DataFrames to merge
        balance_classes: Whether to balance class distribution
        min_samples_per_class: Minimum samples required per class

    Returns:
        Merged DataFrame
    """
    if not datasets:
        raise ValueError("No datasets to merge")

    # Concatenate all datasets
    merged = pd.concat(datasets, ignore_index=True)
    logger.info(f"Merged {len(datasets)} datasets: {len(merged)} total rows")

    if balance_classes:
        # Get class counts
        class_counts = merged['mapped_category'].value_counts()
        logger.info(f"Class distribution before balancing: {dict(class_counts)}")

        # Find minimum class count (but at least min_samples_per_class)
        min_count = max(class_counts.min(), min_samples_per_class)

        # Sample equal number from each class
        balanced_dfs = []
        for category in merged['mapped_category'].unique():
            class_df = merged[merged['mapped_category'] == category]
            if len(class_df) >= min_count:
                balanced_dfs.append(class_df.sample(n=min_count, random_state=42))
            else:
                # If class has fewer samples, use all of them
                balanced_dfs.append(class_df)

        merged = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"After balancing: {len(merged)} rows")

    # Shuffle the merged dataset
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    return merged


def create_training_data(
    sample_per_dataset: Optional[int] = None,
    balance_classes: bool = True,
    save_interim: bool = True
) -> pd.DataFrame:
    """
    Main function to create training data from raw datasets.

    Args:
        sample_per_dataset: Optional sample size per dataset
        balance_classes: Whether to balance class distribution
        save_interim: Whether to save interim files

    Returns:
        Final processed DataFrame ready for training
    """
    logger.info("=" * 50)
    logger.info("Starting data processing pipeline")
    logger.info("=" * 50)

    # Load all available datasets
    loaded_datasets = []

    for filename, config in DATASET_CONFIGS.items():
        filepath = RAW_DIR / filename
        df = load_dataset(
            filepath,
            config['text_column'],
            config['label_column'],
            sample_size=sample_per_dataset
        )
        if df is not None:
            loaded_datasets.append(df)

    if not loaded_datasets:
        logger.warning("No datasets loaded from raw directory")
        logger.info("Creating synthetic dataset for demonstration...")
        df = create_synthetic_dataset()
        loaded_datasets.append(df)

    # Merge raw datasets
    merged_raw = pd.concat(loaded_datasets, ignore_index=True)
    logger.info(f"Total raw data: {len(merged_raw)} rows")

    if save_interim:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        merged_raw.to_csv(INTERIM_DIR / "merged_dataset.csv", index=False)
        logger.info(f"Saved merged dataset to {INTERIM_DIR / 'merged_dataset.csv'}")

    # Clean the data
    cleaned = clean_dataframe(merged_raw, text_column='text', label_column='category')

    # Map labels to unified categories
    mapped = map_labels_in_dataframe(
        cleaned,
        label_column='category',
        text_column='cleaned_text',
        use_keyword_fallback=True
    )

    if save_interim:
        mapped.to_csv(INTERIM_DIR / "cleaned_dataset.csv", index=False)
        logger.info(f"Saved cleaned dataset to {INTERIM_DIR / 'cleaned_dataset.csv'}")

    # Remove outliers
    filtered = remove_outliers(mapped, text_column='cleaned_text')

    # Balance classes if requested
    if balance_classes and len(filtered) > 0:
        class_counts = filtered['mapped_category'].value_counts()
        min_count = min(class_counts.values)

        balanced_dfs = []
        for category in filtered['mapped_category'].unique():
            class_df = filtered[filtered['mapped_category'] == category]
            if len(class_df) > min_count:
                balanced_dfs.append(class_df.sample(n=min_count, random_state=42))
            else:
                balanced_dfs.append(class_df)

        filtered = pd.concat(balanced_dfs, ignore_index=True)
        filtered = filtered.sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare final training data
    final_df = filtered[['cleaned_text', 'mapped_category']].copy()
    final_df.columns = ['text', 'category']

    # Save final processed data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PROCESSED_DIR / "final_training_data.csv", index=False)
    logger.info(f"Saved final training data to {PROCESSED_DIR / 'final_training_data.csv'}")

    logger.info("=" * 50)
    logger.info(f"Data processing complete: {len(final_df)} training samples")
    logger.info("Category distribution:")
    for cat, count in final_df['category'].value_counts().items():
        logger.info(f"  {cat}: {count}")
    logger.info("=" * 50)

    return final_df


def create_synthetic_dataset(samples_per_category: int = 200) -> pd.DataFrame:
    """
    Create a synthetic dataset for demonstration/testing.

    Args:
        samples_per_category: Number of samples per category

    Returns:
        DataFrame with synthetic complaint data
    """
    logger.info("Creating synthetic dataset for demonstration...")

    # Sample complaints for each category
    synthetic_data = {
        "Corruption": [
            "Government official demanded bribe for processing my application",
            "Contractor paid kickback to get the tender approved",
            "Officer asked for money under the table to clear my file",
            "Nepotism in hiring at the municipal office",
            "Misuse of public funds in road construction project",
            "Fraudulent billing by government contractor",
            "Official accepted bribe to overlook violations",
            "Embezzlement of funds meant for welfare scheme",
            "Favoritism shown in awarding government contracts",
            "Illegal commission charged for permit approval",
            "Bribe requested for passport verification",
            "Government employee asking for extra money for service",
            "Corruption in local council office",
            "Funds allocated for school being stolen by officials",
            "Fake vouchers submitted for government travel reimbursement",
            "Officer taking money to hide illegal construction",
            "Zoning laws violated after payment to official",
            "Public land sold illegally by corrupted officials",
            "Police officer asking for bribe to release vehicle",
            "Traffic sergeant demanding money for no reason",
            "Customs officer asking for illegal fee",
            "Tax inspector demanding bribe to reduce assessment",
            "Doctor at government hospital asking for money for treatment",
            "Staff at registrar office asking for bribe for land record",
            "Illegal recruitment process in government department",
            "Funds for poor people were stolen by local representative",
            "Voter list manipulation by election officers",
            "Public distribution system grain being sold in black market",
            "Officer demanding percentage of my contract payment",
            "Illegal sand mining allowed after paying minor officials",
            "Hospital staff charging for free medicines provided by government",
            "School principal asking for donation for admission",
            "Electricity department worker asking for money to fix transformer",
            "Pension officer claiming files are lost unless I pay him",
            "Irrigation water diverted to big farms through bribery",
        ],
        "Utility Issue": [
            "No water supply in our area for the past three days",
            "Frequent power outages affecting our neighborhood",
            "Sewage overflow on the main road near our house",
            "Streetlights not working in our colony for weeks",
            "Garbage not collected from our street regularly",
            "Water pipeline leaking causing road damage",
            "Electricity meter showing wrong readings",
            "Drainage blocked causing flooding during rain",
            "Poor road conditions with multiple potholes",
            "Gas supply interrupted without any notice",
            "Water smells like chemicals in our taps",
            "High voltage causing damage to appliances",
            "Garbage bin overflowing in public park",
            "Open manhole on the sidewalk is dangerous",
            "Potholes on the main highway causing traffic",
            "Broken power line hanging dangerously",
            "Low water pressure making it hard to use",
            "Methane leak from neighborhood landfill",
            "Illegal dumping of waste in residential area",
            "Street flooding after every minor rain",
            "Contaminated water coming from public pump",
            "Incessant load shedding during summer",
            "Dead animals lying on the street not removed",
            "Broken street sign causing confusion",
            "Malfunctioning traffic lights at main crossing",
            "Sewer lines are choked and backing up into basements",
            "Street sweepers never come to our lane",
            "Public tap is broken and wasting tons of water",
            "Illegal construction on storm water drains",
            "Tree branches touching transformers causing sparks",
            "Water tank is never cleaned and full of moss",
            "Stray cattle causing traffic issues on main road",
            "Loud noise from illegal factory in residential zone",
            "Dust pollution from construction site without covers",
            "Toxic waste being dumped into the local pond",
        ],
        "Service Delay": [
            "Passport application pending for over six months",
            "No response to RTI application filed three months ago",
            "Birth certificate not issued despite multiple visits",
            "Driving license renewal taking too long",
            "Property registration delayed without explanation",
            "Pension papers stuck in bureaucratic process",
            "Building permit application pending for a year",
            "No action taken on complaint filed last month",
            "Ration card application still under process",
            "Land records update request ignored",
            "Aadhar card update pending for weeks",
            "Trade license renewal taking forever",
            "Utility connection request pending for months",
            "Delay in processing student scholarship",
            "Health card issuance delayed",
            "No update on application for housing scheme",
            "Marriage certificate registration pending",
            "Long waiting list for government surgery",
            "Delay in response from fire department",
            "Municipal council taking too long for garbage pickup",
            "Inordinate delay in processing industrial permit",
            "File moving too slowly between departments",
            "Application lost in transit between offices",
            "No feedback on previous complaints for weeks",
            "Bureaucratic hurdles in simple document verification",
            "Death certificate application is stuck for 40 days",
            "Freedom fighter pension application hasn't moved for a year",
            "Income certificate verification is pending with tehsildar",
            "Name correction in school records is taking ages",
            "Official at counter says the system is down for a week",
            "Digital signature error preventing file upload for days",
            "Call center for grievances never picks up the phone",
            "Portal is showing processing for the last 90 days",
            "Security clearance for employment is taking forever",
            "Waiting for approval of hospital bill reimbursement",
        ],
        "Harassment": [
            "Supervisor making inappropriate comments at workplace",
            "Discriminated against based on my caste",
            "Facing threats from local goons with political connections",
            "Sexual harassment by senior officer in the department",
            "Bullying and intimidation by colleagues",
            "Verbal abuse from government official during visit",
            "Hostile work environment due to constant criticism",
            "Threatened with termination for filing complaint",
            "Facing discrimination due to my gender",
            "Harassed by inspector during routine check",
            "Online harassment by government employee",
            "Stalking and intimidation at workplace",
            "Casteist slurs used against me in office",
            "Religious discrimination in public service",
            "Forced into overtime without pay through threats",
            "Mental torture by management",
            "Public shaming by official for small error",
            "Inappropriate physical contact during work",
            "Blackmail by someone with power",
            "Abuse of authority to target personal enemies",
            "Harassment of small business owners by officials",
            "Intimidation to withdraw legal case",
            "Persistent harassment from neighbor with police ties",
            "Discriminatory behavior by school staff",
            "Gender-based violence ignored by supervisor",
            "Threatened by landlord who is a government servant",
            "Mental health issues caused by toxic workspace",
            "Denial of promotion due to personal grudge",
            "Public servants mocking my disability",
            "Local leader using thugs to grab my ancestor's land",
            "Constant calls at night from office supervisor",
            "Withholding my salary as a means of control",
            "Refusal of leave for genuine medical emergency as punishment",
            "False allegations made against me to get me fired",
            "Cyberbullying by colleagues on social media groups",
        ],
        "Financial Issue": [
            "Overcharged on electricity bill this month",
            "Bank refusing to process my loan application",
            "Incorrect tax assessment on my property",
            "Pension not credited to account for three months",
            "Unauthorized deductions from my salary",
            "Refund not processed for cancelled service",
            "Hidden fees charged without disclosure",
            "Double billing for water supply",
            "Insurance claim rejected without valid reason",
            "Credit card charged for services not availed",
            "Account hacked and money stolen",
            "Stock market fraud affecting my savings",
            "Unfair interest rate on government loan",
            "Salary delayed for three months",
            "Provident fund withdrawal stuck in system",
            "Incorrect income tax refund amount",
            "GST calculation error on business taxes",
            "Audit findings show financial discrepancy",
            "Bounced check from government department",
            "Fraudulent transaction on debit card",
            "Hidden charges in bank statement",
            "Failure to pay contract workers on time",
            "Discrepancy in pension payments",
            "Unexplained fees for government services",
            "Embezzlement detection in local branch",
            "Subsidies promised are not reaching my account",
            "Drought relief fund not distributed correctly",
            "Crop insurance premium deducted but claim not paid",
            "Bank teller stole money from my fixed deposit",
            "Error in interest calculation for mortgage",
            "Public fund diverted for private functions",
            "Fake accounts used to siphon off welfare money",
            "Audit of the society reveals missing 5 million",
            "Student grant money never arrived in bank",
            "Service tax charged after it was abolished",
        ],
        "Law Enforcement Issue": [
            "Police not registering FIR for theft complaint",
            "No action against repeated domestic violence",
            "Traffic police demanding illegal fine",
            "Emergency response delayed despite multiple calls",
            "Criminal roaming free despite evidence",
            "Police harassment during peaceful protest",
            "No security provided despite threat to life",
            "Hit and run case not being investigated properly",
            "Illegal activities in neighborhood ignored by police",
            "Safety concerns at public place not addressed",
            "Drug dealing in front of police station",
            "Wrongful arrest based on false information",
            "Lack of police patrolling at night",
            "Domestic abuse reported but no police arrival",
            "Witness intimidation not handled by police",
            "Police brutality during questioning",
            "Illegal search and seizure without warrant",
            "Failure to report crime by officer",
            "Traffic laws not enforced on high-speed road",
            "Inadequate security for public events",
            "Illegal weapons found but no investigation",
            "Police taking sides in property dispute",
            "Failure to serve court summons",
            "Corruption within the police department",
            "Missing child case handled poorly",
            "Thieves broke in but police said it was a small matter",
            "Cyber crime department not taking my case seriously",
            "Illegal gambling den running with local police support",
            "Police officer using abusive language with citizens",
            "My stolen phone was tracked but police wont go there",
            "Assault victim being pressurized to reach a settlement",
            "No lady constable present during my arrest at night",
            "Traffic sergeant took my license without a receipt",
            "Failure to investigate suspicious death in my lane",
            "Police vehicle used for personal errands while calls pending",
        ],
    }

    # Generate dataset
    rows = []
    for category, complaints in synthetic_data.items():
        # Repeat and vary complaints to reach desired sample size
        for i in range(samples_per_category):
            base_complaint = complaints[i % len(complaints)]
            # Add more variations and prefixes to improve generalization
            variations = [
                base_complaint,
                f"I want to report: {base_complaint}",
                f"Complaint regarding: {base_complaint}",
                f"Issue: {base_complaint} Please help.",
                f"Urgent: {base_complaint}",
                f"Serious problem: {base_complaint}",
                f"Dear Sir, {base_complaint}",
                f"I am facing a major issue. {base_complaint}",
                f"There is a problem in our colony. {base_complaint}",
                f"Requesting immediate action: {base_complaint}",
            ]
            complaint = variations[i % len(variations)]
            rows.append({
                'text': complaint,
                'category': category,
                'source': 'synthetic'
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Created synthetic dataset with {len(df)} samples")
    return df


if __name__ == "__main__":
    # Run the data processing pipeline
    final_data = create_training_data(
        sample_per_dataset=5000,
        balance_classes=True,
        save_interim=True
    )

    print(f"\nFinal dataset shape: {final_data.shape}")
    print(f"\nSample data:")
    print(final_data.head(10))

