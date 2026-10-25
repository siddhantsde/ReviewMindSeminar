"""Generate a deterministic 1,200-review fictional product intelligence dataset."""
from pathlib import Path
import random
import pandas as pd

SEED = 27
random.seed(SEED)
products = [
    ("Auralis Pulse Headphones", "Electronics", "battery"), ("Auralis Mini Earbuds", "Electronics", "connectivity"),
    ("Nexora Soundbar", "Electronics", "sound"), ("BrightNest Air Fryer", "Home Appliances", "durability"),
    ("BrightNest Kettle", "Home Appliances", "value"), ("Solara Desk Lamp", "Home Appliances", "durability"),
    ("MiraGlow Hair Dryer", "Personal Care", "durability"), ("MiraGlow Skin Brush", "Personal Care", "comfort"),
    ("Vela Smart Watch", "Fashion Accessories", "connectivity"), ("Vela Travel Bag", "Fashion Accessories", "delivery"),
]
issue_phrases = {
    "battery": ["battery drains before the day ends", "charging takes longer than expected", "battery life is excellent on workdays"],
    "connectivity": ["Bluetooth pairing drops during calls", "the connection reconnects without trouble", "setup with my phone was unreliable"],
    "sound": ["dialogue sounds clear at low volume", "bass distorts when volume is high", "microphone sounds muffled in meetings"],
    "durability": ["the hinge still feels sturdy", "the switch stopped working after a month", "the outer material scratches easily"],
    "value": ["the price feels fair for daily use", "it is expensive for the features included", "the value improved after a discount"],
    "comfort": ["the shape stays comfortable for hours", "the handle is tiring to hold", "controls are difficult to reach"],
    "delivery": ["the parcel arrived earlier than expected", "the box arrived crushed and late", "packaging protected the product well"],
}
openers = ["After a few weeks,", "For my daily routine,", "I compared it with an older model and", "On the first weekend,", "During a busy workday,", "My family tried it and", "After setting it up,", "In normal use,", "I bought this for travel and", "One detail stood out:"]
closers = ["I would use it again.", "That matters more than I expected.", "I hope the next version improves this.", "It is a small but noticeable detail.", "The experience has been consistent so far.", "I would check this before recommending it.", "It changed how often I reach for it.", "Overall, the result depends on the task."]
general = {
    "positive": ["easy to set up", "comfortable for longer use", "good value for the price", "well finished and simple to operate", "helpful customer support"],
    "negative": ["customer support took several days to reply", "the controls are confusing at first", "the packaging was missing an accessory", "the material feels lighter than expected", "instructions skipped an important step"],
    "mixed": ["the design is attractive, though one control is awkward", "it works well, but delivery took longer than stated", "the quality is good while the instructions could be clearer", "support resolved the issue after a slow first reply", "useful overall, with one detail that needs improvement"],
}
rows = []
for i in range(1200):
    product, category, signature = products[i % len(products)]
    date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=random.randrange(0, 640))
    neg_rate = .25 if signature == "battery" else .30 if signature == "connectivity" else .20
    if signature == "battery" and date >= pd.Timestamp("2026-08-01"):
        neg_rate = .65
    if signature == "delivery" and date >= pd.Timestamp("2026-08-01"):
        neg_rate = .06
    mixed_rate = .10 if signature == "battery" else .18
    kind = random.choices(["positive", "negative", "mixed"], [1-neg_rate-mixed_rate, neg_rate, mixed_rate])[0]
    if signature == "battery":
        # Keep the defect mentions sparse but explicit, while this demonstration
        # product maintains strong star scores. This deliberately creates an
        # investigation signal rather than equating sentiment and rating.
        phrase = random.choice(["battery drains before the day ends", "charging fails during travel", "battery stopped charging after a few weeks"]) if kind == "negative" else random.choice(general[kind])
    else:
        phrase = random.choice(issue_phrases[signature]) if random.random() < .52 else random.choice(general[kind])
    if kind == "negative":
        text = f"{random.choice(openers)} {phrase}. I am disappointed and would not recommend this experience."
    elif kind == "mixed":
        text = f"{random.choice(openers)} {phrase}, but one improvement is still needed. {random.choice(closers)}"
    else:
        text = f"{random.choice(openers)} {phrase}. {random.choice(closers)}"
    weights = [0, 0, 0, 60, 40] if signature == "battery" and kind == "negative" else [1, 3, 12, 34, 50] if kind == "positive" else [30, 38, 22, 8, 2] if kind == "negative" else [5, 12, 42, 29, 12]
    rating = random.choices([1, 2, 3, 4, 5], weights=weights)[0]
    rows.append({"review_id": f"RM-{i+1:04d}", "product_id": f"P-{i%10+1:02d}", "product_name": product, "category": category, "rating": rating, "review_text": text, "review_date": date.date().isoformat(), "verified_purchase": random.random() < .83, "helpful_votes": random.randrange(36)})
out = Path(__file__).parent / "data" / "sample_reviews.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Wrote {len(rows)} synthetic reviews (seed={SEED}) to {out}")
