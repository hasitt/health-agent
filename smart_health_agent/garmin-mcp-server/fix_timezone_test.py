#!/usr/bin/env python3
"""
Test timezone fix for sleep timestamps
"""

from datetime import datetime, timezone
import pytz

# Your actual data from Aug 6
sleep_start_local = 1754431620000  # Current server shows 01:07 AM
sleep_start_gmt = 1754420820000    # GMT timestamp
sleep_end_local = 1754463420000    # Current server shows 09:57 AM  
sleep_end_gmt = 1754452620000      # GMT timestamp

# Your correct times should be:
# Bedtime: 22:07 (10:07 PM)
# Wake: 06:57 (6:57 AM)

print("🔍 Testing timezone conversion methods...")

print(f"\n1. Current server method (Local timestamp / 1000):")
current_start = datetime.fromtimestamp(sleep_start_local / 1000)
current_end = datetime.fromtimestamp(sleep_end_local / 1000)
print(f"Start: {current_start.strftime('%H:%M')} ({current_start.strftime('%I:%M %p')})")
print(f"End: {current_end.strftime('%H:%M')} ({current_end.strftime('%I:%M %p')})")

print(f"\n2. GMT timestamp + EET conversion:")
# Convert GMT to EET (UTC+2 / UTC+3 for EEST)
eet = pytz.timezone('Europe/Helsinki')  # EET timezone
gmt_start = datetime.fromtimestamp(sleep_start_gmt / 1000, tz=timezone.utc)
gmt_end = datetime.fromtimestamp(sleep_end_gmt / 1000, tz=timezone.utc)
eet_start = gmt_start.astimezone(eet)
eet_end = gmt_end.astimezone(eet)
print(f"Start: {eet_start.strftime('%H:%M')} ({eet_start.strftime('%I:%M %p')})")
print(f"End: {eet_end.strftime('%H:%M')} ({eet_end.strftime('%I:%M %p')})")

print(f"\n3. Alternative: Local timestamp with timezone adjustment:")
# Maybe the local timestamp is already in local time but we need different conversion
alt_start = datetime.fromtimestamp(sleep_start_local / 1000)
alt_end = datetime.fromtimestamp(sleep_end_local / 1000)
# Subtract 3 hours (potential timezone offset issue)
from datetime import timedelta
adj_start = alt_start - timedelta(hours=3)
adj_end = alt_end - timedelta(hours=3)
print(f"Start: {adj_start.strftime('%H:%M')} ({adj_start.strftime('%I:%M %p')})")
print(f"End: {adj_end.strftime('%H:%M')} ({adj_end.strftime('%I:%M %p')})")

print(f"\n4. Direct UTC timestamp conversion:")
utc_start = datetime.fromtimestamp(sleep_start_gmt / 1000, tz=timezone.utc)
utc_end = datetime.fromtimestamp(sleep_end_gmt / 1000, tz=timezone.utc)
print(f"UTC Start: {utc_start.strftime('%H:%M')} ({utc_start.strftime('%I:%M %p')})")
print(f"UTC End: {utc_end.strftime('%H:%M')} ({utc_end.strftime('%I:%M %p')})")

print(f"\n✅ Expected correct times:")
print(f"Bedtime should be: 22:07 (10:07 PM)")
print(f"Wake time should be: 06:57 (6:57 AM)")

print(f"\n🎯 Method 2 (GMT + EET conversion) appears to match your correct times!")