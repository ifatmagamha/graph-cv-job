import os
import random
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------
# 1) Global settings
# -----------------------------
N_CVS = 300
N_JOBS = 90

OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

DOMAINS = {
    "data": ["python", "sql", "machine learning", "statistics", "pandas", "numpy", "data visualization", "etl", "power bi", "tableau"],
    "software": ["python", "java", "javascript", "git", "docker", "rest api", "microservices", "linux", "testing", "cloud"],
    "cybersecurity": ["network security", "linux", "siem", "incident response", "penetration testing", "cryptography", "firewalls", "risk assessment"],
    "finance": ["excel", "financial modeling", "accounting", "risk analysis", "forecasting", "powerpoint", "sql", "business intelligence"],
    "marketing": ["seo", "content marketing", "google analytics", "social media", "copywriting", "crm", "email marketing", "branding"],
    "design": ["figma", "ui design", "ux research", "typography", "adobe photoshop", "adobe illustrator", "wireframing", "design systems"],
    "management": ["project management", "agile", "scrum", "stakeholder management", "communication", "planning", "budgeting", "leadership"],
}

# Some cross-domain skills to add realism
CROSS_SKILLS = ["communication", "teamwork", "problem solving", "english", "reporting", "presentation"]

LEVELS = ["junior", "intermediate", "senior"]

# Define level difficulty mapping (for matching)
LEVEL_RANK = {"junior": 0, "intermediate": 1, "senior": 2}

# -----------------------------
# 2) Helper functions
# -----------------------------
def sample_domain():
    return random.choice(list(DOMAINS.keys()))

def sample_level():
    # more juniors than seniors (realistic)
    return random.choices(LEVELS, weights=[0.5, 0.35, 0.15])[0]

def sample_profile_type():
    return random.choices(["specialized", "polyvalent"], weights=[0.65, 0.35])[0]

def make_skills(domain: str, profile_type: str, k_base: int):
    core = DOMAINS[domain]
    if profile_type == "specialized":
        skills = random.sample(core, k=min(k_base, len(core)))
        # add 1-2 cross skills
        skills += random.sample(CROSS_SKILLS, k=random.choice([1, 2]))
    else:
        # polyvalent: mix across 2-3 domains
        domains_mix = random.sample(list(DOMAINS.keys()), k=random.choice([2, 3]))
        mix_pool = []
        for d in domains_mix:
            mix_pool += DOMAINS[d]
        mix_pool = list(set(mix_pool))
        skills = random.sample(mix_pool, k=min(k_base + 2, len(mix_pool)))
        skills += random.sample(CROSS_SKILLS, k=random.choice([2, 3]))
    # unique & stable order
    skills = list(dict.fromkeys(skills))
    return skills

def cv_text_template(domain, level, skills):
    bullets = "\n- " + "\n- ".join(skills[:8])
    return (
        f"CV Profile: {level} candidate in {domain}. "
        f"Experience with relevant tools and projects. Skills:{bullets}\n"
        f"Education: Bachelor's / Master's related to {domain}. "
        f"Soft skills: communication, teamwork."
    )

def job_text_template(domain, level_required, skills):
    bullets = "\n- " + "\n- ".join(skills[:8])
    return (
        f"Job Offer: {level_required} position in {domain}. "
        f"We are looking for a candidate with the following requirements:{bullets}\n"
        f"Nice to have: reporting, presentation. Location: Hybrid."
    )

def match_score(cv_row, job_row):
    cv_sk = set(cv_row["skills_list"])
    job_sk = set(job_row["skills_list"])
    overlap = len(cv_sk.intersection(job_sk))
    union = len(cv_sk.union(job_sk))
    jaccard = overlap / union if union else 0.0

    # level compatibility: CV level should be >= required ideally
    lv_cv = LEVEL_RANK[cv_row["level"]]
    lv_job = LEVEL_RANK[job_row["level_required"]]
    level_bonus = 0.15 if lv_cv >= lv_job else -0.10

    # domain bonus if same
    domain_bonus = 0.10 if cv_row["domain"] == job_row["domain"] else 0.0

    return jaccard + level_bonus + domain_bonus

# -----------------------------
# 3) Generate CVs
# -----------------------------
cv_rows = []
for i in range(N_CVS):
    domain = sample_domain()
    level = sample_level()
    profile_type = sample_profile_type()

    # more skills for seniors
    k_base = {"junior": 6, "intermediate": 8, "senior": 10}[level]
    skills = make_skills(domain, profile_type, k_base=k_base)

    cv_id = f"CV_{i:04d}"
    cv_rows.append({
        "cv_id": cv_id,
        "text": cv_text_template(domain, level, skills),
        "skills": ", ".join(skills),
        "level": level,
        "profile_type": profile_type,
        "domain": domain,
    })

cv_df = pd.DataFrame(cv_rows)

# -----------------------------
# 4) Generate Jobs
# -----------------------------
job_rows = []
for j in range(N_JOBS):
    domain = sample_domain()
    level_required = random.choices(LEVELS, weights=[0.45, 0.40, 0.15])[0]

    # job skill requirements by level
    k_req = {"junior": 6, "intermediate": 8, "senior": 10}[level_required]
    # jobs are usually more "specialized" in one domain
    skills = make_skills(domain, "specialized", k_base=k_req)

    job_id = f"JOB_{j:04d}"
    job_rows.append({
        "job_id": job_id,
        "text": job_text_template(domain, level_required, skills),
        "skills": ", ".join(skills),
        "level_required": level_required,
        "domain": domain,
    })

job_df = pd.DataFrame(job_rows)

# Parse skills to list for scoring
cv_df["skills_list"] = cv_df["skills"].apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
job_df["skills_list"] = job_df["skills"].apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])

# -----------------------------
# 5) Generate matches (edges)
# -----------------------------
# We create "known matches" by scoring CV-JOB pairs and sampling top ones.
edges = []
top_k_per_cv = 4  # average known matches per CV (controls graph density)

job_records = job_df.to_dict(orient="records")

for _, cv in cv_df.iterrows():
    # score against all jobs (can be heavy if huge; ok for N_JOBS ~ 100)
    scored = []
    cv_row = cv.to_dict()
    for job in job_records:
        s = match_score(cv_row, job)
        scored.append((job["job_id"], s))

    scored.sort(key=lambda x: x[1], reverse=True)

    # pick top K with a bit of randomness to avoid perfect matches only
    candidates = scored[:15]  # consider top 15 as plausible
    chosen = random.sample(candidates, k=min(top_k_per_cv, len(candidates)))

    for job_id, s in chosen:
        edges.append({"cv_id": cv_row["cv_id"], "job_id": job_id, "label": 1, "score_hint": round(s, 4)})

edges_df = pd.DataFrame(edges).drop_duplicates(subset=["cv_id", "job_id"])

# -----------------------------
# 6) Save CSVs
# -----------------------------
# Drop helper skills_list columns before saving
cv_out = cv_df.drop(columns=["skills_list"])
job_out = job_df.drop(columns=["skills_list"])

cv_out.to_csv(os.path.join(OUT_DIR, "cv_raw.csv"), index=False)
job_out.to_csv(os.path.join(OUT_DIR, "job_raw.csv"), index=False)
edges_df[["cv_id", "job_id", "label"]].to_csv(os.path.join(OUT_DIR, "matches_raw.csv"), index=False)

print("✅ Synthetic datasets generated:")
print(f"- {OUT_DIR}/cv_raw.csv   (rows={len(cv_out)})")
print(f"- {OUT_DIR}/job_raw.csv  (rows={len(job_out)})")
print(f"- {OUT_DIR}/matches_raw.csv (rows={len(edges_df)})")

print("\nSample CV:")
print(cv_out.head(2))
print("\nSample Job:")
print(job_out.head(2))
print("\nSample Matches:")
print(edges_df.head(5))