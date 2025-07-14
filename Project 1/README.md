# Weather Dashboard (Python Script)

This project provides a simple command-line weather dashboard using the WeatherAPI.com service. It fetches the current weather for any city and displays a bar chart of temperature, humidity, and pressure.

## Features
- Fetches current weather data for any city using WeatherAPI.com
- Visualizes temperature (°C), humidity (%), and pressure (mb) in a bar chart
- Simple to use: just run the script and enter a city name

## Requirements
- Python 3.x
- The following Python packages:
  - requests
  - matplotlib
  - seaborn

You can install the required packages with:
```sh
pip install requests matplotlib seaborn
```

## Usage
1. Make sure you have Python 3 installed.
2. Install the required packages (see above).
3. The API key is already set in the script (`weather_dashboard.py`).
4. Run the script:
   ```sh
   python weather_dashboard.py
   ```
5. Enter the city name when prompted.
6. A bar chart will appear showing the current temperature, humidity, and pressure for the city.

## API Key
- The script uses a WeatherAPI.com API key, which is already set in the code. If you want to use your own key, edit the `API_KEY` variable in `weather_dashboard.py`.

## Notes
- This script uses the current weather endpoint, not the forecast.
- No Flask or web server is required; this is a standalone Python script.

---

If you have any questions or want to extend the functionality, feel free to ask! 