import streamlit as st
import joblib 
import pandas as pd
import xgboost

scaler = joblib.load("models/scaler.h5")
Model = joblib.load("models/model.h5")
Inputs=['Depth (m)', 'Surface WOB Avg(klb)', 'RPM Surface Avg (rpm)',
       'Torque Abs Avg (f-b)', 'SPP Avg (psig)', 'Flow in Pum Avg(gpm)',
       'M.wt(ppg)', 'HOB(hr)', 'Rev On Bit (krev)', 'Bit Diameter', 'TFA(in2)',
       'pit volume(bbl)', 'PP(ppg)', 'OVB(ppg)']

def predict(Depth,Surface_WOB_Avg,RPM_Surface_Avg,Torque,SPP_Avg,Flow_Pum_Avg,Mud_wt,HOB,Rev_On_Bit,Bit_Diameter,TFA,pit_volume,PP, OVB):
    test_df = pd.DataFrame(columns = Inputs,index=[0])
    test_df.at[0,"Depth (m)"] = Depth
    test_df.at[0,"Surface WOB Avg(klb)"] = Surface_WOB_Avg
    test_df.at[0,"RPM Surface Avg (rpm)"] = RPM_Surface_Avg
    test_df.at[0,"Torque Abs Avg (f-b)"] = Torque
    test_df.at[0,"SPP Avg (psig)"] = SPP_Avg
    test_df.at[0,"Flow in Pum Avg(gpm)"] = Flow_Pum_Avg
    test_df.at[0,"M.wt(ppg)"] = Mud_wt
    test_df.at[0,"HOB(hr)"] = HOB
    test_df.at[0,"Rev On Bit (krev)"] = Rev_On_Bit
    test_df.at[0,"Bit Diameter"] = Bit_Diameter
    test_df.at[0,"TFA(in2)"] = TFA
    test_df.at[0,"pit volume(bbl)"] = pit_volume
    test_df.at[0,"PP(ppg)"] = PP
    test_df.at[0,"OVB(ppg)"] = OVB

    result = Model.predict(scaler.transform(test_df))[0]
    return result
    

def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)    

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;"> ROP ML prediction App</h1> 
    </div> 
   """
     
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
    # following lines create boxes in which user can enter data required to make prediction
    st.sidebar.title("Choose your Features") 
    Depth = st.sidebar.slider("Total drilling Depth", min_value=0, max_value=100000, value=0, step=1)
    Surface_WOB_Avg = st.sidebar.number_input("Surface WOB Avg(klb)")
    RPM_Surface_Avg = st.sidebar.number_input("RPM Surface Avg (rpm)")
    Torque = st.sidebar.number_input('Torque Abs Avg (f-b)')
    SPP_Avg = st.sidebar.number_input('SPP Avg (psig)')
    Flow_Pum_Avg = st.sidebar.number_input('Flow in Pum Avg(gpm)')
    Mud_wt = st.sidebar.number_input('M.wt(ppg)')
    HOB = st.sidebar.number_input('HOB(hr)')
    Rev_On_Bit = st.sidebar.number_input('Rev On Bit (krev)')
    Bit_Diameter = st.sidebar.number_input('Bit Diameter')
    TFA = st.sidebar.number_input('TFA(in2)')
    pit_volume = st.sidebar.number_input('pit volume(bbl)')
    PP = st.sidebar.number_input('PP(ppg)')
    OVB = st.sidebar.number_input('OVB(ppg)')
    result =""
          
    # when 'Predict' is clicked, make the prediction and store it 
    if st.sidebar.button("Predict"): 
        result = predict(Depth,Surface_WOB_Avg,RPM_Surface_Avg,Torque,SPP_Avg,Flow_Pum_Avg,Mud_wt,HOB,Rev_On_Bit,Bit_Diameter,TFA,pit_volume,PP, OVB)
                
        ## Print final Prediction 
        st.markdown(f'<h1 style="color:#33ff33;font-size:40px;text-align:center;border-style: solid;border-width:5px;border-color:#fbff00;">{result}</h1>', unsafe_allow_html=True)
   
   ## show resturant image
    st.image('R.jfif')    
     
if __name__=='__main__': 
    main()
