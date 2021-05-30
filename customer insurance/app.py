

import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

def predict_insurance(age,salery):
    input=np.array([[age, salery]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():

    st.title("CUSTOMER INSURANCE")
    
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Customer Insurance Prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    age = st.text_input("age")
    salery = st.text_input("salery")

    not_interested_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:white;text-align:center;"> Customer Will Not Be Buying Insurance</h2>
       </div>
    """
    
    interested_html="""  
      <div style="background-color:#025246;padding:10px >
       <h2 style="color:black ;text-align:center;"> Customer Will Be Buying Insurance </h2>
       </div>
    """

    if st.button("Predict"):
        output = predict_insurance(age, salery)
        st.success('The Probability Of Customer Buying Insurance Is : {}'.format(output))

        if output > 0.5:
            st.markdown(interested_html,unsafe_allow_html=True)
        else:
            st.markdown(not_interested_html,unsafe_allow_html=True)



if __name__ =='__main__':
    main()

