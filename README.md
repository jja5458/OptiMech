# OptiMech: Mechanical System Optimization for Energy Efficiency

## Project Overview
OptiMech is a project aimed at optimizing mechanical systems in a university commercial building to reduce energy consumption by 25%. The focus of this project is on the building's HVAC system, which provides heating and cooling throughout the year using a direct expansion system (for cooling) and a hot water system (for heating). 

By analyzing historical energy consumption data and operational parameters, this project aims to develop a simulation model that can predict energy usage and identify optimal configurations for minimizing energy consumption.

## Project Scope
The scope of this project includes the following key areas:

1. **Data Collection and Preprocessing:**
   - Collect historical energy consumption data for the HVAC system.
   - Preprocess the data to handle missing values, normalize values, and create time-based features such as day of the week, month, and season.

2. **Feature Engineering:**
   - Create new features based on observations that better represent the underlying patterns in the data. This includes features such as:
     - Time-based features: Day of the week, month, season.
     - Weather-related features: Temperature difference between indoor and outdoor conditions, outdoor temperature bins.
     - Lag features: Previous day's energy consumption.
     - Rolling statistics: 7-day rolling mean and standard deviation of energy consumption.
     - Interaction features: Energy consumption vs. temperature difference.
   
3. **Simulation and Modeling:**
   - Develop simulation models to predict energy consumption based on various operational parameters and environmental conditions.
   - Use the simulation results to identify configurations that lead to optimal energy consumption.
   
4. **Optimization:**
   - Implement optimization algorithms that use the simulation models to recommend the most efficient settings for the HVAC system.
   
5. **User Interface:**
   - Build a simple user interface (if applicable) to allow stakeholders to input operational parameters, view simulation results, and receive optimization recommendations.

6. **Evaluation and Results:**
   - Evaluate the performance of the models and simulations in predicting energy consumption.
   - Analyze the potential energy savings from optimization, with the goal of reducing energy usage by 25%.

## Technologies Used
- **Python**: Programming language used for data processing, modeling, and simulations.
- **TensorFlow**: For building predictive models based on historical data.
- **PyTorch**: Used for training machine learning models (if applicable).
- **SciPy**: For optimization tasks and solving numerical problems in simulations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For visualizing data and simulation results.
- **Seaborn**: For enhanced data visualization.

## Installation
1. Clone the repository:
2. Create a virtual environment:
3. Activate the virtual environment:
- On Windows:
  ```
  .\venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```
4. Install the required dependencies:


## Running the Project
1. **Preprocess and Engineer Features:**
To run the feature engineering script and preprocess the data, execute:

2. **Simulate Energy Consumption:**
To run the simulation models and generate predictions, execute:

3. **Optimization and Results:**
For optimization tasks based on simulation results, execute:


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- Thanks to the open-source community for the libraries and tools used in this project.

