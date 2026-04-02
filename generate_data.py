"""
Script to generate a realistic synthetic customer dataset for Blys AI assessment.
Run once: python generate_data.py
"""

import random
import csv
from datetime import datetime, timedelta

random.seed(42)

SERVICES = [
    "Massage", "Facial", "Wellness Package", "Deep Tissue Massage",
    "Hot Stone Massage", "Couples Massage", "Aromatherapy", "Pedicure",
    "Manicure", "Body Scrub", "Reflexology", "Swedish Massage",
]

POSITIVE_REVIEWS = [
    "Absolutely loved it, will book again!",
    "Highly recommend, therapist was excellent.",
    "Best massage I've ever had.",
    "Very professional and relaxing experience.",
    "Worth every dollar, incredible service.",
    "Fantastic, arrived on time and very skilled.",
    "Exceeded my expectations completely.",
    "Perfect stress relief, thank you!",
]

NEGATIVE_REVIEWS = [
    "Too expensive for what you get.",
    "Therapist arrived late and seemed rushed.",
    "Not worth the price, disappointing.",
    "Expected more for the cost.",
    "Service was okay but nothing special.",
    "Had to reschedule twice, very inconvenient.",
    "Quality has dropped compared to last time.",
]

NEUTRAL_REVIEWS = [
    "Decent service, met expectations.",
    "Fine, nothing outstanding but okay.",
    "Average experience, might try again.",
    "Service was standard.",
    "Okay overall, some improvements needed.",
    "",
]


def random_date(start_days_ago=365, end_days_ago=1):
    delta = random.randint(end_days_ago, start_days_ago)
    return (datetime.now() - timedelta(days=delta)).strftime("%Y-%m-%d")


def pick_review(freq, spending):
    if freq >= 7 or spending >= 250:
        pool = POSITIVE_REVIEWS * 3 + NEUTRAL_REVIEWS
    elif freq <= 2 or spending <= 70:
        pool = NEGATIVE_REVIEWS * 2 + NEUTRAL_REVIEWS + POSITIVE_REVIEWS
    else:
        pool = NEUTRAL_REVIEWS + POSITIVE_REVIEWS + NEGATIVE_REVIEWS
    return random.choice(pool)


rows = []
for i in range(1, 501):
    customer_id = 1000 + i
    segment = random.choices(
        ["champion", "loyal", "at_risk", "new", "lost"],
        weights=[0.15, 0.25, 0.25, 0.20, 0.15],
    )[0]

    if segment == "champion":
        freq = random.randint(8, 20)
        spend = round(random.uniform(250, 600), 2)
        last = random_date(30, 1)
    elif segment == "loyal":
        freq = random.randint(5, 10)
        spend = round(random.uniform(150, 350), 2)
        last = random_date(60, 5)
    elif segment == "at_risk":
        freq = random.randint(3, 7)
        spend = round(random.uniform(80, 200), 2)
        last = random_date(180, 61)
    elif segment == "new":
        freq = random.randint(1, 3)
        spend = round(random.uniform(50, 150), 2)
        last = random_date(30, 1)
    else:
        freq = random.randint(1, 3)
        spend = round(random.uniform(30, 90), 2)
        last = random_date(365, 181)

    service = random.choice(SERVICES)
    review = pick_review(freq, spend)

    # Inject some missing values (~5%)
    if random.random() < 0.05:
        review = ""
    if random.random() < 0.03:
        spend = ""
    if random.random() < 0.02:
        freq = ""

    rows.append([customer_id, freq, spend, service, review, last])

with open("data/customers.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Customer_ID", "Booking_Frequency", "Avg_Spending",
        "Preferred_Service", "Review_Text", "Last_Activity",
    ])
    writer.writerows(rows)

print(f"Generated {len(rows)} customer records → data/customers.csv")
