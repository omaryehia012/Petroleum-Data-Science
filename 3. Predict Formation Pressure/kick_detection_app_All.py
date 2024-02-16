import streamlit as st
import joblib 
import pandas as pd
import xgboost

scaler = joblib.load("models/scaler.h5")
Model = joblib.load("model_All.h5")
Inputs=['TVD(ft)', 'BITSIZE(in)', 'NPHI(%)',
       'Corrected Bulk Density(gm/cc)', 'Deep Resistivity (Ohm)', 'ROP(M/hr)',
       'WOB(KLb)', 'RPM', 'Torque(lb.F)', 'Stand Pipe Pressure(Psi)',
       'Flow In(GPM)','Temp - Out', 'Total Gas(PPM)']

def predict(TVD_FT,BITSIZE,NPHI,Corrected_Bulk_Density,Deep_Resistivity ,ROP,WOB,RPM,Torque,Stand_Pipe_Pressure,
                  Flow_In,Temp_Out,Total_Gas):
    test_df = pd.DataFrame(columns = Inputs,index=[0])
    test_df.at[0,"TVD(ft)"] = TVD_FT
    test_df.at[0,"BITSIZE(in)"] = BITSIZE
    test_df.at[0,"NPHI(%)"] = NPHI
    test_df.at[0,"Corrected Bulk Density(gm/cc)"] = Corrected_Bulk_Density
    test_df.at[0,"Deep Resistivity (Ohm)"] = Deep_Resistivity
    test_df.at[0,"ROP(M/hr)"] = ROP
    test_df.at[0,"WOB(KLb)"] = WOB
    test_df.at[0,"RPM"] = RPM
    test_df.at[0,"Torque(lb.F)"] = Torque
    test_df.at[0,"Stand Pipe Pressure(Psi)"] = Stand_Pipe_Pressure
    test_df.at[0,"Flow In(GPM)"] = Flow_In
    #test_df.at[0,"Temp - In"] = Temp_In
    test_df.at[0,"Temp - Out"] = Temp_Out
    test_df.at[0,"Total Gas(PPM)"] = Total_Gas

    result = Model.predict(scaler.transform(test_df))[0]
    return result
    

def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)    

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;"> Kick Detection System ML prediction App</h1> 
    </div> 
   """
    
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
    # following lines create boxes in which user can enter data required to make prediction
    st.sidebar.title("Choose your Features")
    TVD_FT = st.sidebar.number_input('TVD(ft)', min_value=0, max_value=100000, value=0, step=1)
    BITSIZE = st.sidebar.number_input('BITSIZE(in)')
    NPHI = st.sidebar.number_input('NPHI(%)')
    Corrected_Bulk_Density = st.sidebar.number_input('Corrected Bulk Density(gm/cc)')
    Deep_Resistivity = st.sidebar.number_input('Deep Resistivity (Ohm)')
    ROP = st.sidebar.number_input('ROP(M/hr)')
    WOB = st.sidebar.number_input('WOB(KLb)')
    RPM = st.sidebar.number_input('RPM')
    Torque = st.sidebar.number_input('Torque(lb.F)')
    Stand_Pipe_Pressure = st.sidebar.number_input('Stand Pipe Pressure(Psi)')
    Flow_In = st.sidebar.number_input('Flow In(GPM)')
    #Temp_In = st.sidebar.number_input('Temp - In')
    Temp_Out = st.sidebar.number_input('Temp - Out')
    Total_Gas = st.sidebar.number_input('Total Gas(PPM)')
    result =""
          
    # when 'Predict' is clicked, make the prediction and store it 
    if st.sidebar.button("Predict"): 
        result = predict(TVD_FT,BITSIZE,NPHI,Corrected_Bulk_Density,Deep_Resistivity ,ROP,WOB,RPM,Torque,Stand_Pipe_Pressure,
                  Flow_In,Temp_Out,Total_Gas)
                
        ## Print final Prediction 
        st.markdown(f'<h1 style="color:#33ff33;font-size:40px;text-align:center;border-style: solid;border-width:5px;border-color:#fbff00;">{result}</h1>', unsafe_allow_html=True)
   
   ## show resturant image
    st.image('R.jfif')    
     
if __name__=='__main__': 
    main()
