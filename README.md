# Drone Motor Anomaly Detection Visualization

![alt text](https://github.com/Jared-Kirschbaum/time-series-anomaly-detection/blob/main/fig.png)

## Overview

This project simulates time series data for four drone motors over a period of 5 years, applies anomaly detection using a rolling average and Z-score method, and creates interactive visualizations with Plotly. The resulting dashboard is saved as an HTML file that you can view in any modern web browser.

## Features

- **Data Simulation:**  
  Generates realistic test data combining annual seasonal variations, weekly operational cycles, motor-specific frequency variations, slight trends, and random noise.

- **Anomaly Detection:**  
  Implements a rolling window approach to smooth data and detect anomalies based on a Z-score threshold. The script optimizes the window size to match an expected anomaly proportion.

- **Interactive Visualizations:**  
  Uses Plotly to create interactive, modern, and customizable plots. The final visualization is saved as `fig.html`.

- **Customizable Settings:**  
  Easily adjust data simulation parameters, anomaly detection thresholds, and visualization styles.

## Requirements

- Python 3.6 or higher
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Plotly](https://plotly.com/python/)
- (Optional) [Kaleido](https://github.com/plotly/Kaleido) for saving static images

You can install the required packages using pip:

```bash
pip install pandas numpy plotly kaleido
```
Usage
Clone the Repository:

bash
Copy
git clone https://github.com/yourusername/drone-motor-anomaly-detection.git
cd drone-motor-anomaly-detection
Run the Script:

Execute the main Python script:

bash
Copy
python time-series.py
The script will:

Generate 5 years of simulated data for four drone motors.
Perform anomaly detection with an optimized rolling window.
Create interactive visualizations using Plotly.
Save the final interactive figure as fig.html in the project directory.
View the Results:

Open fig.html in your web browser to explore the interactive plots.

Project Structure
```
├── time-series.py    # Main script
├── time-series.ipynb # testing notebook for examples.
└── README.md         # Project documentation.
```
## Data Simulation:
Modify the generate_test_data() function in time-series.py to change seasonal patterns, trends, noise levels, or simulation periods.

## Anomaly Detection:
Adjust parameters such as the z_threshold and the window size in the optimize_window_size() function to fine-tune anomaly detection.

## Visualization:
Change the color scheme in the colors dictionary or update Plotly layout options in the plot_motor_data() function to match your preferred style.

## License
This project is licensed under the MIT License.

## Acknowledgements
This project utilizes powerful open-source libraries including Pandas, NumPy, and Plotly to demonstrate a complete workflow for simulating time series data, performing anomaly detection, and building interactive visualizations.
