# Onboarding Application

Welcome to the Onboarding Application! This application helps manage onboarding processes for new employees.

## Setup Instructions

Follow these steps to set up the application:

### 1. Fill in Your Database Credentials

In the `onboarding.py` file, locate the following line:
engine = create_engine('mysql+pymysql://root:...@localhost/onboarding')
Replace root with your MySQL username and ... with your MySQL password. Ensure that the database name (onboarding in this case) matches your MySQL database name.

### 2. Set Up a Virtual Environment

First, make sure you have Python installed on your system. Then, follow these steps to create and activate a virtual environment:

### Create a virtual environment
python -m venv venv

### Activate the virtual environment
### On Windows
venv\Scripts\activate
### On macOS/Linux
source venv/bin/activate

### 3. Install Required Packages
Once your virtual environment is activated, install the required packages listed in requirements.txt using pip:
pip install -r requirements.txt
This command will install all the necessary dependencies for the Onboarding Application.

### 4. Run the Application
You're all set! You can now run the Onboarding Application:
python onboarding.py
