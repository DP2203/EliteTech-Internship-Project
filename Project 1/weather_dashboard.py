import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Set your WeatherAPI.com API key here
API_KEY = 'f89460f8fabe4d99866131208250207'
BASE_URL = "http://api.weatherapi.com/v1/current.json"
CITY = input("Enter city name: ")
params = {
    "key": API_KEY,
    "q": CITY
}

response = requests.get(BASE_URL, params=params)

if response.status_code != 200:
    print("Failed to fetch data:", response.json())
    exit()

data = response.json()

# Extract relevant fields
temperature = data['current']['temp_c']
humidity = data['current']['humidity']
pressure = data['current'].get('pressure_mb', None)  # May not always be present
condition = data['current']['condition']['text'] if 'condition' in data['current'] else 'N/A'
precip_mm = data['current'].get('precip_mm', None)  # Precipitation in mm (proxy for rain)

# Print a simple, human-readable summary
print(f"\nWeather in {CITY}:")
print(f"  Condition : {condition}")
print(f"  Temperature: {temperature} °C")
print(f"  Humidity   : {humidity} %")
if pressure is not None:
    print(f"  Pressure   : {pressure} mb")
if precip_mm is not None:
    max_precip = 50  # Arbitrary high value for precipitation in mm
    rain_pct = (precip_mm / max_precip) * 100
    print(f"  Rain       : {rain_pct:.1f}% (precipitation: {precip_mm} mm)")

# Calculate and print percentages
max_temp = 50  # Typical max temperature in °C
max_humidity = 100  # Max humidity in %
max_pressure = 1100  # Typical max pressure in mb
max_precip = 50  # Arbitrary high value for precipitation in mm

temp_pct = (temperature / max_temp) * 100
humidity_pct = (humidity / max_humidity) * 100
pressure_pct = (pressure / max_pressure) * 100 if pressure is not None else None
precip_pct = (precip_mm / max_precip) * 100 if precip_mm is not None else None

print("\nPercentages relative to typical maximums:")
print(f"  Temperature: {temp_pct:.1f}% of 50°C")
print(f"  Humidity   : {humidity_pct:.1f}% of 100%")
if pressure_pct is not None:
    print(f"  Pressure   : {pressure_pct:.1f}% of 1100 mb")
if precip_pct is not None:
    print(f"  Rain       : {precip_pct:.1f}% of 50 mm")

# Prepare data for plotting
metrics = ['Temperature (°C)', 'Humidity (%)']
values = [temperature, humidity]
labels = [f'{temperature} °C', f'{humidity} %']
colors = ['red', 'blue']

if pressure is not None:
    metrics.append('Pressure (mb)')
    values.append(pressure)
    labels.append(f'{pressure} mb')
    colors.append('green')

if precip_mm is not None:
    rain_pct = (precip_mm / max_precip) * 100
    metrics.append('Rain (%)')
    values.append(rain_pct)
    labels.append(f'{rain_pct:.1f}%')
    colors.append('deepskyblue')

sns.set_theme(style="darkgrid")
plt.figure(figsize=(8, 6))
barplot = sns.barplot(x=metrics, y=values, palette=colors)
plt.title(f"Current Weather in {CITY}\nCondition: {condition}")
plt.ylabel("Value")
plt.xlabel("Weather Metric")

# Add value labels above each bar
for i, (bar, label) in enumerate(zip(barplot.patches, labels)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.02, label,
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show() 