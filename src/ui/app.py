
import requests
import streamlit as st

from src.config import get_settings



def start_services():
    settings=get_settings()
    
    st.title("Jira AI Automation Dashboard")
    
    # SECTION 1: Analyze Ticket API (Requires Input)
    st.header("Ticket analysis")
    with st.container(border=True):
        summary=st.text_input("Summary")
        description=st.text_area("Description")

        if st.button("Run AI analysis"):
            payload={"summary":summary,"description":description}
            with st.spinner("Running analysis..."):

                # call analyze api
                res=requests.post(f"{settings.api_url}/api/v1/analyze",json=payload)
                print(res.json())
                if res.status_code==200:
                    data=res.json()
                    st.success(f"Predicted Priority: {data['predicted_priority']}")
                    st.write(f"Recommended Dev: {data['recommended_dev']}")
                    st.json(data['dev_mood']) 
                else:
                    st.error("Analysis failed.")
        else:
            st.warning("Please enter both summary and description.")



if __name__ == "__main__":
    start_services()
