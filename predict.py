import joblib
import pandas as pd

# Load the trained model
model = joblib.load("house_price_model.pkl")

print("\n==============================")
print("      HOUSE PRICE PREDICTOR")
print("==============================")

# Get the features used during training
features = model.feature_names_in_

# Create a new house with all required features
house = pd.DataFrame(0, index=[0], columns=features)

# Get information from the user
overall_qual = int(input("Overall Quality (1-10): "))
gr_liv_area = int(input("Living Area (sq ft): "))
garage_cars = int(input("Number of Garage Cars: "))
total_bsmt_sf = int(input("Basement Area (sq ft): "))
full_bath = int(input("Number of Full Bathrooms: "))
year_built = int(input("Year Built: "))

# Set the values
house["OverallQual"] = overall_qual
house["GrLivArea"] = gr_liv_area
house["GarageCars"] = garage_cars
house["TotalBsmtSF"] = total_bsmt_sf
house["FullBath"] = full_bath
house["YearBuilt"] = year_built

# Make prediction
prediction = model.predict(house)

print("\n------------------------------")
print(f"Predicted House Price: ${prediction[0]:,.2f}")
print("------------------------------")