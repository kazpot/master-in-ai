from datetime import datetime
import pytz

utc = pytz.utc
jst = pytz.timezone("Asia/Tokyo")

now = datetime.now(tz=utc)
jst_now = now.astimezone(jst)

print(now)
print(jst_now)