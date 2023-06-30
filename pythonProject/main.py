import numpy as np
import pandas as pd
from flask import Flask,render_template,request,redirect
import pickle
from flask_cors import CORS,cross_origin #Cross-Origin Resource Sharing

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))

data=pd.read_csv("output.csv")
map_CompanyName = {'Company 1': 0, 'Company 10': 1, 'Company 11': 2, 'Company 12': 3, 'Company 25': 4, 'Company 26': 5,
                   'Company 27': 6, 'Company 28': 7, 'Company 29': 8, 'Company 3': 9, 'Company 4': 10, 'Company 6': 11,
                   'Company 7': 12, 'Company 8': 13, 'Company 9': 14}
map_City = {'Austin': 0, 'Chicago': 1, 'Dallas': 2, 'Denver': 3, 'El Paso': 4, 'Houston': 5, 'Los Angeles': 6,
            'Miami': 7, 'Ohio': 8, 'San Antonio': 9, 'San Jose': 10, 'Tucson': 11}
map_Salesperson = {'Antman': 0, 'Beast': 1, 'Heman': 2, 'Hulk': 3, 'Kattapa': 4, 'Magneto': 5, 'Superman': 6, 'Thor': 7}
map_Region = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
map_ProductName = {'Apple Jam': 0, 'Basil ': 1, 'Bread': 2, 'Brown Rice': 3, 'Burger': 4, 'Butter': 5, 'Cabbage': 6,
                   'Cake': 7, 'Cheese': 8, 'Chocolate': 9, 'Coconuts': 10, 'Coffee': 11, 'Coke': 12, 'Eclairs': 13,
                   'Fudge': 14, 'Green Tea': 15, 'Musturd Oil': 16, 'Pineapple Jam': 17, 'Pizza': 18, 'Rice': 19,
                   'Soup': 20, 'Tea': 21, 'Tomotao ketchup': 22, 'Wine': 23}
map_Category = {'Category 1': 0, 'Category 10': 1, 'Category 11': 2, 'Category 12': 3, 'Category 13': 4,
                'Category 14': 5, 'Category 2': 6, 'Category 3': 7, 'Category 4': 8, 'Category 5': 9, 'Category 6': 10,
                'Category 7': 11, 'Category 8': 12, 'Category 9': 13}

@app.route('/',methods=['GET','POST'])
def index():
    list_CompanyName=sorted(list(map_CompanyName.keys()))
    list_City=sorted(list(map_City.keys()))
    list_Salesperson = sorted(list(map_Salesperson.keys()))
    list_Region = sorted(list(map_Region.keys()))
    list_ProductName = sorted(list(map_ProductName.keys()))
    list_Category = sorted(list(map_Category.keys()))
    list_Unit_Price=sorted(data['Unit Price'].unique())
    list_Quantity=sorted(data['Quantity'].unique())
    list_hour=sorted(data['hour'].unique())
    list_day=sorted(data['day'].unique())
    list_month=sorted(data['month'].unique())
    list_year=sorted(data['year'].unique())
    list_weekday=sorted(data['weekday'].unique())
    return render_template('index.html',list_CompanyName=list_CompanyName,
                           list_City=list_City,
                           list_Salesperson=list_Salesperson,
                           list_Region=list_Region,
                           list_ProductName=list_ProductName,
                           list_Category=list_Category,
                           list_Unit_Price=list_Unit_Price,
                           list_Quantity=list_Quantity,
                           list_hour=list_hour,
                           list_day=list_day,
                           list_month=list_month,
                           list_year=list_year,
                           list_weekday=list_weekday
                           )
# Unit Price 	Quantity 	Revenue 	hour 	day 	month 	year 	weekday
@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    company=request.form.get('company')
    city=request.form.get('city')
    salesperson = request.form.get('salesperson')
    Region = request.form.get('Region')
    ProductName = request.form.get('ProductName')
    Unit_Price = request.form.get('list_Unit_Price')
    Quantity = request.form.get('Quantity')
    category = request.form.get('Category')
    hour = request.form.get('hour')
    day = request.form.get('day')
    month = request.form.get('month')
    year = request.form.get('year')
    weekday = request.form.get('weekday')
    new_company=map_CompanyName.get(company)
    new_city=map_City.get(city)
    new_sales=map_Salesperson.get(salesperson)
    new_region=map_Region.get(Region)
    new_product=map_ProductName.get(ProductName)
    new_category=map_Category.get(category)
    print(new_company, new_city, new_sales, new_region, new_product, new_category, Unit_Price, Quantity, hour, day,
          month, year, weekday)
    print(Region)
    prediction=model.predict(pd.DataFrame(columns=['Company Name', 'City', 'Salesperson', 'Region',
       'Product Name', 'Category', 'Unit Price', 'Quantity', 'hour',
       'day', 'month', 'year', 'weekday'],data=np.array([new_company,new_city,new_sales,new_region,new_product,new_category,Unit_Price,Quantity,hour,day,month,year,weekday]).reshape(1,13)))
    print(company,city,salesperson,Region,ProductName,category,Unit_Price,Quantity,hour,day,month,year,weekday)
    print(prediction)
    print(new_company,new_city,new_sales,new_region,new_product,new_category,Unit_Price,Quantity,hour,day,month,year,weekday)
    return str(np.round(prediction[0],2))


if __name__=="__main__":
    app.run(debug=True)
