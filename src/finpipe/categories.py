"""Fixed category taxonomy.

The LLM categorizer must answer with one of these values, and the output is
validated against this enum, so a hallucinated category can never reach gold.
"""

from enum import Enum


class Category(str, Enum):
    GROCERIES = "Groceries"
    DINING = "Dining"
    TRANSPORTATION = "Transportation"
    FUEL = "Fuel"
    UTILITIES = "Utilities"
    SUBSCRIPTIONS = "Subscriptions"
    SHOPPING = "Shopping"
    ENTERTAINMENT = "Entertainment"
    HEALTH = "Health"
    TRAVEL = "Travel"
    HOME = "Home"
    DOGS = "Dogs"
    MORTGAGE = "Mortgage"
    STUDENTLOANS = "Student Loans"
    INTEREST = "Interest"
    INTERESTCHARGE = "Interest Charge"
    DONATION = "Donation"
    INSURANCE = "Insurance"
    INCOME = "Income"
    TRANSFER = "Transfer"
    OTHER = "Other"


CATEGORY_VALUES = [c.value for c in Category]
