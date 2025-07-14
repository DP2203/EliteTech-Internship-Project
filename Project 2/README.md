# Weather Report Generator

This project generates a comprehensive PDF weather report from a CSV file containing weather data for various cities.

## Features
- Reads weather data from `weather_data.csv`
- Calculates summary statistics (average, min, max for temperature, humidity, wind speed)
- Generates a formatted PDF report (`Weather_Report_Unique.pdf`)
- Includes a bar chart of temperature by city
- Visually appealing layout with colored headers and date

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install matplotlib
   ```
2. Ensure your weather data is in `weather_data.csv` with columns:
   - City, Temperature, Humidity, Wind Speed

## Usage
Run the script:
```
python weather_report_pdf.py
```

The PDF report will be generated as `Weather_Report_Unique.pdf` in the same directory. 