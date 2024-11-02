# Superconductor Critical Temperature model

The main app can be launched from the file "main_app.py"
Doing so will launch a Gradio app on http://127.0.0.1:7860, or whatever the default address is set to.
A link to the app will also be printed to the console.

The application runs a random forest regression model. The user can choose from a list of sample data points, and 
predict the critical temperature for them based on the data. These sample points are not included in the training data 
for the model.

The files used for analyzing and cleaning data as well as for fine-tuning models can be found in the "Testing and Data 
Management" folder. The data as it is stored in the original database can be found in the "Data" folder. A version of 
the data that has been cleaned of outliers and normalized can be found in the "Cleaned Data" folder.
