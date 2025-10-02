# Input Output Tables

The Welsh Government Data Science Unit and Input Output Analysis team have collaborated to build an Input Output Table Analysis Tool.

This application can be accessed online via Streamlit: 

Alternatively, you may choose to clone the repository and run the application on your own device. This may be useful for users looking to keep their data on their device, rather than upload to Streamlit.IO.

__Please note: This tool uses a fictional Input-Output table for a four-sector economy by default. This data is provided for illustrative purposes only. To use your own Input-Output table, select the option to “Download the current IOT to use as a template”, insert your data, and re-upload the file__

## Dependencies

1. Install our _iot-leontief-python_ package:

a. Clone the repo
```
git clone https://github.com/wgdsu/iot-leontief-python.git
```

b. Switch to the project repo:
```
cd "iot-leontief-python"
```

c. Install dependencies
```
pip install -r requirements.txt
```

d. Install our package
```
pip install .
```
## Clone this tool
2. Clone this application:
``` 
git clone https://github.com/wgdsu/InputOutputAnalysisTool.git

cd InputOutputModellingTool
```

3. Install additional Python dependencies
```
pip install -r requirements.txt
```

## Running the application

In the root directory `/InputOutputModellingTool/` run:
```
python -m streamlit run app.py
```


