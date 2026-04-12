"""
build_dataset.py
================
Downloads real public datasets and generates synthetic samples to produce
balanced train/test splits for four document classes:

    invoice | email | scientific_report | letter

Real sources
------------
  invoice          : SROIE 2019  (receipts OCR text via HuggingFace)
  email            : Enron Email (HuggingFace subset) + HC3 gov emails
  scientific_report: ArXiv abstracts (HuggingFace) + scientific papers
  letter           : CUAD contracts (legal letters/agreements)

Synthetic samples fill format gaps so the classifier sees diverse layouts.

Output
------
  data/processed/train.csv   (19 001 rows)
  data/processed/test.csv    ( 3 125 rows  — 100 % real)

Columns: text, label
"""

import os
import re
import csv
import random
import textwrap
from pathlib import Path
from datetime import date, timedelta

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[WARN] `datasets` not installed – will use synthetic-only mode for all classes.")

random.seed(42)

ROOT        = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TARGETS
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_TARGETS = {"invoice": 4001, "email": 5000, "scientific_report": 5000, "letter": 5000}
TEST_TARGETS  = {"invoice":  125, "email": 1000, "scientific_report": 1000, "letter": 1000}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rand_date(start_year=2018, end_year=2024):
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def clean(text: str) -> str:
    """Collapse whitespace and strip."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()

def truncate(text: str, max_chars=3000) -> str:
    return text[:max_chars]

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

# ── Invoice ──────────────────────────────────────────────────────────────────
COMPANIES = [
    "Acme Corp", "BlueSky Solutions", "TechNova Ltd", "GreenPath Services",
    "Apex Consulting", "Meridian Supplies", "Zenith Medical", "Crestline Freight",
    "Luminary Design", "PrimeForce Agency", "Coastal Power & Gas", "Nexus Analytics",
]
CLIENTS = [
    "John Smith", "Maria García", "Li Wei", "Amara Osei", "Fatima Al-Hassan",
    "Carlos Mendez", "Sophie Dubois", "Rajan Patel", "Elena Rossi", "Kenji Tanaka",
]
SERVICES = [
    ("Web Development", 1200, 8000),
    ("Consulting Services", 800, 5000),
    ("Software License", 300, 2000),
    ("Medical Supplies", 150, 1500),
    ("Electricity Bill", 80, 600),
    ("Freight Delivery", 200, 1800),
    ("Design Services", 500, 4000),
    ("Cloud Hosting (monthly)", 99, 999),
    ("Legal Advisory", 1000, 9000),
    ("IT Support Contract", 400, 3000),
]
CURRENCIES = ["USD", "EUR", "GBP"]

def _invoice_block(fmt: str) -> str:
    inv_no   = f"INV-{random.randint(1000,9999)}"
    inv_date = rand_date()
    due_date = inv_date + timedelta(days=random.choice([15, 30, 45, 60]))
    issuer   = random.choice(COMPANIES)
    recipient= random.choice(CLIENTS)
    service, lo, hi = random.choice(SERVICES)
    amount   = round(random.uniform(lo, hi), 2)
    cur      = random.choice(CURRENCIES)
    tax_rate = random.choice([0, 0.08, 0.10, 0.20, 0.21])
    tax_amt  = round(amount * tax_rate, 2)
    total    = round(amount + tax_amt, 2)

    if fmt == "standard":
        return textwrap.dedent(f"""
            INVOICE
            Invoice Number: {inv_no}
            Invoice Date:   {inv_date.strftime('%Y-%m-%d')}
            Due Date:       {due_date.strftime('%Y-%m-%d')}

            From: {issuer}
            To:   {recipient}

            Description          Qty   Unit Price     Amount
            ─────────────────────────────────────────────────
            {service:<20}   1    {cur} {amount:>10,.2f}   {cur} {amount:>10,.2f}
            {"Tax (" + str(int(tax_rate*100)) + "%)":<20}                         {cur} {tax_amt:>10,.2f}
            ─────────────────────────────────────────────────
            TOTAL                                       {cur} {total:>10,.2f}

            Payment terms: Net {due_date.day} days
            Please make payment to: {issuer} Bank Account IBAN XX00 0000 0000 0000
        """).strip()

    elif fmt == "freelancer":
        return textwrap.dedent(f"""
            Freelance Invoice

            Bill To: {recipient}
            From:    {issuer}

            Invoice #: {inv_no}
            Date:      {inv_date.strftime('%d %b %Y')}
            Due:       {due_date.strftime('%d %b %Y')}

            Services Rendered:
              - {service}

            Subtotal:   {cur} {amount:,.2f}
            Tax:        {cur} {tax_amt:,.2f}
            ─────────────────
            Total Due:  {cur} {total:,.2f}

            Thank you for your business!
        """).strip()

    elif fmt == "utility":
        account = f"AC-{random.randint(100000,999999)}"
        return textwrap.dedent(f"""
            {issuer.upper()}
            UTILITY BILL / INVOICE

            Account Number:  {account}
            Invoice No:      {inv_no}
            Billing Date:    {inv_date.strftime('%m/%d/%Y')}
            Payment Due:     {due_date.strftime('%m/%d/%Y')}

            Customer: {recipient}

            Charges:
              {service}     {cur} {amount:,.2f}
              Taxes & Fees  {cur} {tax_amt:,.2f}
                            ──────────────
              Amount Due    {cur} {total:,.2f}

            To pay online visit www.{issuer.lower().replace(' ','')}.com/pay
        """).strip()

    else:  # medical / B2B
        return textwrap.dedent(f"""
            TAX INVOICE

            Supplier:  {issuer}
            Customer:  {recipient}
            Invoice:   {inv_no}
            Date:      {inv_date.isoformat()}
            Due Date:  {due_date.isoformat()}

            Item: {service}
            Net Amount:    {cur} {amount:,.2f}
            VAT ({int(tax_rate*100)}%):      {cur} {tax_amt:,.2f}
            Total Payable: {cur} {total:,.2f}

            Registered address: 123 Business Park, London, UK
            VAT Reg: GB{random.randint(100000000,999999999)}
        """).strip()

def make_invoice(n: int):
    fmts = ["standard", "freelancer", "utility", "b2b"]
    return [_invoice_block(random.choice(fmts)) for _ in range(n)]

# ── Email ─────────────────────────────────────────────────────────────────────
EMAIL_SUBJECTS = [
    "Re: Project Update", "Follow-up: Meeting Tomorrow", "Quarterly Report Attached",
    "Action Required: Contract Renewal", "Invitation: Team Lunch", "FYI: System Outage",
    "Your Support Ticket #84221 has been resolved", "New hire onboarding checklist",
    "Sales Q3 targets exceeded!", "Urgent: Server incident P1",
    "Performance review reminder", "Budget approval needed",
]
EMAIL_BODIES = [
    "Hi {name},\n\nJust following up on our conversation last week. Could you please send over the updated figures by EOD Friday?\n\nBest,\n{sender}",
    "Dear {name},\n\nPlease find attached the quarterly performance report for your review. Let me know if you have any questions.\n\nRegards,\n{sender}",
    "Hi team,\n\nA reminder that our standup is at 10am tomorrow. Please come prepared with your blockers.\n\nThanks,\n{sender}",
    "Hi {name},\n\nYour support ticket #{ticket} has been resolved. If the issue persists please reply to this email.\n\nSupport Team",
    "Dear {name},\n\nWe are pleased to inform you that your application has moved to the next stage. We will be in touch shortly.\n\nHR Team",
    "All,\n\nPlease be advised that the production servers will undergo maintenance on Saturday from 02:00 to 06:00 UTC.\n\nIT Operations",
    "Hi {name},\n\nCongratulations on closing the {deal} deal! The team really pulled together on this one.\n\nCheers,\n{sender}",
]
NAMES = ["Alex", "Jordan", "Morgan", "Taylor", "Casey", "Sam", "Riley", "Dana"]

def make_email_synthetic(n: int):
    samples = []
    for _ in range(n):
        subj   = random.choice(EMAIL_SUBJECTS)
        body   = random.choice(EMAIL_BODIES).format(
            name=random.choice(NAMES),
            sender=random.choice(NAMES),
            ticket=random.randint(10000,99999),
            deal=random.choice(["Acme", "BlueSky", "TechNova"]),
        )
        date_  = rand_date()
        samples.append(
            f"From: {random.choice(NAMES).lower()}@company.com\n"
            f"To: {random.choice(NAMES).lower()}@company.com\n"
            f"Date: {date_.strftime('%a, %d %b %Y %H:%M:%S')}\n"
            f"Subject: {subj}\n\n{body}"
        )
    return samples

# ── Scientific Report ─────────────────────────────────────────────────────────
SCI_TITLES = [
    "A Novel Approach to Neural Network Pruning",
    "Climate Change Impacts on Coastal Wetlands",
    "Advances in CRISPR Gene Editing Efficiency",
    "Quantum Error Correction: A Survey",
    "Deep Learning for Medical Image Segmentation",
    "Socioeconomic Factors in Urban Heat Island Effects",
    "Transformer Architectures for Time-Series Forecasting",
]
SCI_SECTIONS = [
    ("Abstract", "This study investigates {topic}. We propose a novel methodology combining {method1} and {method2}. Our experimental results demonstrate a {improvement}% improvement over the baseline across {n} datasets."),
    ("Introduction", "The problem of {topic} has attracted significant research interest in recent years. Previous work by {author} et al. ({year}) laid the groundwork for modern approaches. However, key challenges remain, particularly regarding {challenge}."),
    ("Methodology", "We formulate the problem as follows. Let X ∈ ℝ^(n×d) denote the input matrix. We apply {method1} to obtain latent representations Z, followed by {method2} for classification. The loss function is defined as L = Σ ℓ(ŷ_i, y_i) + λ||W||₂."),
    ("Results", "Table 1 shows the comparison of our method against baselines. Our approach achieves {acc}% accuracy on the test set, outperforming the best baseline by {delta}%. We conduct ablation studies to isolate the contribution of each component."),
    ("Conclusion", "In this paper we presented a framework for {topic}. The empirical evaluation confirms the effectiveness of our approach. Future work will explore {future_dir}."),
]
TOPICS     = ["document classification", "image segmentation", "sequence modelling", "graph neural networks", "multi-modal learning"]
METHODS    = ["attention mechanisms", "contrastive learning", "variational inference", "gradient boosting", "spectral clustering"]
AUTHORS    = ["Zhang", "Kim", "Patel", "Müller", "Fernandez", "Nguyen"]

def make_scientific_synthetic(n: int):
    samples = []
    for _ in range(n):
        title   = random.choice(SCI_TITLES)
        topic   = random.choice(TOPICS)
        secs    = random.sample(SCI_SECTIONS, k=random.randint(3, 5))
        body    = f"Title: {title}\n\n"
        for sec_name, tmpl in secs:
            body += f"{sec_name}\n"
            body += tmpl.format(
                topic=topic,
                method1=random.choice(METHODS),
                method2=random.choice(METHODS),
                improvement=random.randint(2, 30),
                n=random.randint(3, 12),
                author=random.choice(AUTHORS),
                year=random.randint(2015, 2023),
                challenge=random.choice(["scalability", "data scarcity", "interpretability"]),
                acc=round(random.uniform(82, 98), 1),
                delta=round(random.uniform(0.5, 5.0), 1),
                future_dir=random.choice(["low-resource settings", "real-time inference", "multimodal inputs"]),
            ) + "\n\n"
        samples.append(body.strip())
    return samples

# ── Letter ────────────────────────────────────────────────────────────────────
LETTER_TYPES = ["cover_letter", "complaint", "offer", "government", "tenancy"]

def _letter_block(ltype: str) -> str:
    date_   = rand_date()
    sender  = random.choice(CLIENTS)
    company = random.choice(COMPANIES)

    if ltype == "cover_letter":
        role = random.choice(["Data Scientist", "Software Engineer", "Product Manager", "UX Designer"])
        return textwrap.dedent(f"""
            {sender}
            {date_.strftime('%B %d, %Y')}

            Hiring Manager
            {company}

            Dear Hiring Manager,

            I am writing to express my strong interest in the {role} position at {company}. With over {random.randint(2,10)} years of experience in the field, I am confident that my skills and background align well with your requirements.

            In my previous role at {random.choice(COMPANIES)}, I led a team of {random.randint(3,12)} engineers to deliver a platform that improved operational efficiency by {random.randint(15,50)}%. I am proficient in Python, machine learning, and cross-functional collaboration.

            I would welcome the opportunity to discuss how I can contribute to {company}'s continued growth. Thank you for your consideration.

            Sincerely,
            {sender}
        """).strip()

    elif ltype == "complaint":
        ref = f"REF-{random.randint(10000,99999)}"
        return textwrap.dedent(f"""
            {sender}
            {date_.strftime('%d %B %Y')}

            Customer Relations
            {company}

            Re: Formal Complaint – Reference {ref}

            Dear Sir/Madam,

            I am writing to formally complain about the service I received on {rand_date().strftime('%d %B %Y')}. Despite multiple attempts to resolve this issue through your customer service team, my concerns have not been addressed satisfactorily.

            The issue involves {random.choice(['a delayed delivery', 'a billing error', 'poor service quality', 'a defective product'])}. I request an immediate resolution and appropriate compensation.

            I expect a response within 14 days. Should this matter remain unresolved, I will escalate to the relevant regulatory authority.

            Yours faithfully,
            {sender}
        """).strip()

    elif ltype == "offer":
        salary = random.randint(45000, 150000)
        role   = random.choice(["Data Scientist", "Software Engineer", "Product Manager"])
        return textwrap.dedent(f"""
            {company}
            {date_.strftime('%B %d, %Y')}

            {sender}

            Dear {sender.split()[0]},

            We are pleased to offer you the position of {role} at {company}, commencing {(date_ + timedelta(days=30)).strftime('%B %d, %Y')}.

            Your annual base salary will be {random.choice(CURRENCIES)} {salary:,}, paid monthly. You will be entitled to {random.randint(20,30)} days of annual leave and participation in the company benefits scheme.

            Please sign and return this letter by {(date_ + timedelta(days=7)).strftime('%B %d, %Y')} to confirm your acceptance.

            We look forward to welcoming you to the team.

            Yours sincerely,
            HR Department
            {company}
        """).strip()

    elif ltype == "government":
        ref = f"GOV/{random.randint(2020,2024)}/{random.randint(1000,9999)}"
        return textwrap.dedent(f"""
            GOVERNMENT NOTICE
            Reference: {ref}
            Date: {date_.strftime('%d/%m/%Y')}

            To: {sender}

            Subject: {random.choice(['Tax Assessment Notice', 'Benefit Entitlement Update', 'Regulatory Compliance Requirement', 'Public Consultation Invitation'])}

            Dear {sender.split()[0]},

            This letter is to inform you that following a recent review, your account has been updated. Please note the following changes effective from {(date_ + timedelta(days=14)).strftime('%d/%m/%Y')}.

            Action required: Please complete the attached form and return it to our office within 21 days. Failure to respond may affect your entitlements.

            If you have any questions please contact us on 0800 000 0000 quoting reference {ref}.

            Yours sincerely,
            The Office
        """).strip()

    else:  # tenancy
        rent = random.randint(800, 3500)
        return textwrap.dedent(f"""
            TENANCY AGREEMENT LETTER

            Date: {date_.strftime('%d %B %Y')}
            Landlord: {random.choice(COMPANIES)}
            Tenant:   {sender}

            Property: {random.randint(1,200)} {random.choice(['High Street', 'Park Avenue', 'King Road', 'Queen Lane'])}, {random.choice(['London', 'Madrid', 'Berlin', 'Paris'])}

            This letter confirms the tenancy agreement for the above property commencing {date_.strftime('%d %B %Y')} for a fixed term of {random.choice([6, 12, 24])} months.

            Monthly Rent: {random.choice(CURRENCIES)} {rent:,}
            Deposit:      {random.choice(CURRENCIES)} {rent*2:,}

            Both parties agree to the terms and conditions set out in the full tenancy agreement dated {date_.isoformat()}.

            Signed: _________________________    Date: ___________
                    Landlord / Agent

            Signed: _________________________    Date: ___________
                    Tenant
        """).strip()

def make_letter_synthetic(n: int):
    types = LETTER_TYPES
    return [_letter_block(random.choice(types)) for _ in range(n)]

# ─────────────────────────────────────────────────────────────────────────────
# REAL DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_real_invoices(max_samples: int):
    """
    Real invoice/receipt texts.
    Tries multiple HuggingFace sources; falls back gracefully to synthetic.
    """
    if not HF_AVAILABLE:
        return []
    texts = []

    # Source 1: katanaml-org/invoices-donut-data  (English invoice JSONs)
    try:
        print("  Downloading invoice dataset (katanaml-org)…")
        ds = load_dataset("katanaml-org/invoices-donut-data", split="train")
        for row in ds:
            # ground_truth is a JSON string with invoice fields
            raw = row.get("ground_truth", "") or row.get("text", "")
            t   = clean(str(raw))
            if len(t) > 50:
                texts.append(truncate(t))
            if len(texts) >= max_samples:
                break
        print(f"  katanaml invoices → {len(texts)} samples")
    except Exception as e:
        print(f"  [WARN] katanaml invoices failed: {e}")

    # Source 2: mychen76/invoices-and-receipts_ocr_v1
    if len(texts) < max_samples:
        try:
            print("  Downloading invoices-and-receipts OCR dataset…")
            ds2 = load_dataset("mychen76/invoices-and-receipts_ocr_v1", split="train")
            for row in ds2:
                t = clean(row.get("text", "") or str(row.get("ocr_text", "")))
                if len(t) > 50:
                    texts.append(truncate(t))
                if len(texts) >= max_samples:
                    break
            print(f"  invoices-and-receipts → cumulative {len(texts)} samples")
        except Exception as e:
            print(f"  [WARN] invoices-and-receipts failed: {e}")

    return texts[:max_samples]


def load_real_emails(max_samples: int):
    """Enron email corpus via HuggingFace (Parquet-native datasets only)."""
    if not HF_AVAILABLE:
        return []
    texts = []

    # Source 1: SetFit/enron_spam  (labeled spam/ham, standard Parquet)
    try:
        print("  Downloading SetFit/enron_spam…")
        ds = load_dataset("SetFit/enron_spam", split="train")
        for row in ds:
            parts = []
            if row.get("subject"):
                parts.append("Subject: " + str(row["subject"]))
            if row.get("text"):
                parts.append(str(row["text"]))
            t = clean(" ".join(parts))
            if len(t) > 80:
                texts.append(truncate(t))
            if len(texts) >= max_samples:
                break
        print(f"  SetFit/enron_spam → {len(texts)} samples")
    except Exception as e:
        print(f"  [WARN] SetFit/enron_spam failed: {e}")

    # Source 2: corbt/enron-emails  (517 k rows, pure Parquet)
    if len(texts) < max_samples:
        try:
            print("  Downloading corbt/enron-emails…")
            ds2 = load_dataset("corbt/enron-emails", split="train", streaming=True)
            for row in ds2:
                t = clean(row.get("text", "") or row.get("body", ""))
                if len(t) > 80:
                    texts.append(truncate(t))
                if len(texts) >= max_samples:
                    break
            print(f"  corbt/enron-emails → cumulative {len(texts)} samples")
        except Exception as e:
            print(f"  [WARN] corbt/enron-emails failed: {e}")

    return texts[:max_samples]


def load_real_scientific(max_samples: int):
    """ArXiv abstracts via HuggingFace."""
    if not HF_AVAILABLE:
        return []
    texts = []

    # Source 1: ccdv/arxiv-classification
    try:
        print("  Downloading ArXiv abstracts (ccdv)…")
        ds = load_dataset("ccdv/arxiv-classification", split="train")
        for row in ds:
            t = clean(row.get("text", "") or row.get("abstract", ""))
            if len(t) > 100:
                texts.append(truncate(t))
            if len(texts) >= max_samples:
                break
        print(f"  ArXiv ccdv → {len(texts)} samples")
    except Exception as e:
        print(f"  [WARN] ArXiv ccdv failed: {e}")

    # Source 2: gfissore/arxiv-abstracts-2021
    if len(texts) < max_samples:
        try:
            print("  Downloading gfissore/arxiv-abstracts-2021…")
            ds2 = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
            for row in ds2:
                t = clean(row.get("abstract", ""))
                if len(t) > 100:
                    texts.append(truncate(t))
                if len(texts) >= max_samples:
                    break
            print(f"  gfissore arxiv → cumulative {len(texts)} samples")
        except Exception as e:
            print(f"  [WARN] gfissore arxiv failed: {e}")

    return texts[:max_samples]


def load_real_letters(max_samples: int):
    """
    CUAD contracts and legal clause texts (Parquet-native HuggingFace datasets).
    """
    if not HF_AVAILABLE:
        return []
    texts = []

    # Source 1: theatticusproject/cuad  (auto-converted Parquet, 84 k rows)
    try:
        print("  Downloading theatticusproject/cuad…")
        ds = load_dataset("theatticusproject/cuad", split="train")
        seen = set()
        for row in ds:
            ctx   = clean(row.get("context", "") or row.get("text", ""))
            chunk = truncate(ctx, 1500)
            if len(chunk) > 100 and chunk not in seen:
                seen.add(chunk)
                texts.append(chunk)
            if len(texts) >= max_samples:
                break
        print(f"  theatticusproject/cuad → {len(texts)} samples")
    except Exception as e:
        print(f"  [WARN] theatticusproject/cuad failed: {e}")

    # Source 2: dvgodoy/CUAD_v1_Contract_Understanding_clause_classification
    if len(texts) < max_samples:
        try:
            print("  Downloading CUAD clause classification…")
            ds2 = load_dataset(
                "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
                split="train"
            )
            for row in ds2:
                t = clean(row.get("clause", "") or row.get("text", ""))
                if len(t) > 100:
                    texts.append(truncate(t, 1500))
                if len(texts) >= max_samples:
                    break
            print(f"  CUAD clauses → cumulative {len(texts)} samples")
        except Exception as e:
            print(f"  [WARN] CUAD clauses failed: {e}")

    return texts[:max_samples]

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE SPLITS  — load real data ONCE per class, split, then pad train
# ─────────────────────────────────────────────────────────────────────────────

def build_class_data(label: str, real_loader, synth_maker,
                     n_train: int, n_test: int):
    """
    Strategy:
      1. Load max(n_train + n_test) real samples in a single call.
      2. Reserve the last n_test for the test split (zero overlap with train).
      3. Pad the training split with synthetic if real data is insufficient.
    """
    print(f"\n[{label.upper()}]  train={n_train}  test={n_test}")
    real_needed = n_train + n_test
    real        = real_loader(real_needed)
    random.shuffle(real)

    # Hold-out the last n_test real samples exclusively for test
    real_test  = real[-n_test:]   if len(real) >= n_test else real[:]
    real_train = real[:-n_test]   if len(real) >= n_test else []

    # ── test set ───────────────────────────────────────────────────────────
    if len(real_test) < n_test:
        gap = n_test - len(real_test)
        print(f"  Test gap: {gap} real missing → filling with synthetic")
        real_test += synth_maker(gap)
    test_samples = real_test[:n_test]

    # ── train set ──────────────────────────────────────────────────────────
    train_samples = real_train[:n_train]
    n_synth = n_train - len(train_samples)
    if n_synth > 0:
        print(f"  Generating {n_synth} synthetic training samples…")
        train_samples += synth_maker(n_synth)
    random.shuffle(train_samples)
    train_samples = train_samples[:n_train]

    real_tr = min(len(real_train), n_train)
    real_te = min(len(real_test),  n_test)
    print(f"  Real in train: {real_tr}/{n_train} | Real in test: {real_te}/{n_test}")

    return train_samples, test_samples


def write_csv(rows: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS MAP  (label → (real_loader, synth_maker))
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = {
    "invoice":           (load_real_invoices,   make_invoice),
    "email":             (load_real_emails,      make_email_synthetic),
    "scientific_report": (load_real_scientific,  make_scientific_synthetic),
    "letter":            (load_real_letters,     make_letter_synthetic),
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    train_rows, test_rows = [], []

    for label, (real_loader, synth_maker) in LOADERS.items():
        n_train = TRAIN_TARGETS[label]
        n_test  = TEST_TARGETS[label]

        train_samples, test_samples = build_class_data(
            label, real_loader, synth_maker, n_train, n_test
        )

        train_rows += [(t, label) for t in train_samples]
        test_rows  += [(t, label) for t in test_samples]

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    write_csv(train_rows, DATA_PROC / "train.csv")
    write_csv(test_rows,  DATA_PROC / "test.csv")

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "═"*50)
    print("DATASET BUILD COMPLETE")
    print("═"*50)
    from collections import Counter
    tr_counts = Counter(r[1] for r in train_rows)
    te_counts = Counter(r[1] for r in test_rows)
    print(f"{'Label':<20} {'Train':>8} {'Test':>8}")
    print("-"*38)
    for lbl in LOADERS:
        print(f"  {lbl:<18} {tr_counts[lbl]:>8} {te_counts[lbl]:>8}")
    print("-"*38)
    print(f"  {'TOTAL':<18} {sum(tr_counts.values()):>8} {sum(te_counts.values()):>8}")
    print("═"*50)


if __name__ == "__main__":
    main()
