import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title='Agusan del Sur | Palay Intelligence',page_icon='🌾',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif} .stApp{background:#09131e;color:#e9f1f7} [data-testid="stSidebar"]{background:#101f2d;border-right:1px solid #294052} h1,h2,h3{font-family:'Outfit',sans-serif;letter-spacing:-.035em} h1{color:#f2faf7!important} [data-testid="stMetric"]{background:linear-gradient(135deg,#142a36,#10202e);border:1px solid #294354;border-radius:17px;padding:19px} [data-testid="stMetricLabel"]{color:#a5bfcc} [data-testid="stMetricValue"]{color:#f6fbff} div.stButton>button{border-radius:10px} .hero{padding:25px 30px;border:1px solid #2c524d;border-radius:20px;background:linear-gradient(115deg,#173d3b,#102431 70%);margin-bottom:20px}.eyebrow{color:#79dbc0;font-size:12px;letter-spacing:2px;font-weight:800}.hero h1{margin:7px 0;font-size:39px}.muted{color:#aac1cb}.section{font-size:12px;color:#6ee0bb;font-weight:800;letter-spacing:1.7px;margin:12px 0}.insight{padding:16px 19px;background:#122735;border-left:3px solid #5ed6af;border-radius:9px;color:#c5d9e3;margin:10px 0 20px}.small{color:#8da7b5;font-size:12px}</style>''',unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Annual series transcribed from final Chapters 4–5, Table 1.
    annual = pd.DataFrame({
        'Year':[2021,2022,2023,2024,2025],
        'Production':[270963.35,232758,288638,306775,326394],
        'Harvested Area':[83634,80358,82211,87039,83555],
        'Yield':[3.24,2.90,3.51,3.52,3.91],
        'Farmgate Price':[14.96,15.28,18.73,19.76,14.9325],
        'Rainfall':[3729.0,4494.4,3615.9,2919.1,3553.6],
        'Temperature':[26.15,25.80,26.25,26.65,25.85]
    })
    return annual

a=load_data()
metrics=['Production','Harvested Area','Yield','Farmgate Price']
units={'Production':'metric tons','Harvested Area':'hectares','Yield':'mt/ha','Farmgate Price':'₱/kg','Rainfall':'mm','Temperature':'°C'}
colors={'Production':'#5ee0b0','Harvested Area':'#60a5fa','Yield':'#f7c36d','Farmgate Price':'#fb8f8f','Rainfall':'#6ab8ff','Temperature':'#f7b267'}

def style(fig,height=380):
    fig.update_layout(template='plotly_dark',height=height,paper_bgcolor='#10212e',plot_bgcolor='#10212e',font=dict(color='#c8d8e1',family='DM Sans'),margin=dict(l=22,r=20,t=45,b=35),legend=dict(orientation='h',y=1.15),hovermode='x unified')
    fig.update_xaxes(showgrid=False,dtick=1 if fig.layout.xaxis.type!='category' else None)
    fig.update_yaxes(gridcolor='#263c49',zerolinecolor='#3b5360')
    return fig

def fit(col):
    x=a[['Year']].values;y=a[col].values
    model=LinearRegression().fit(x,y)
    pred=model.predict(x)
    years=np.arange(2026,2031)
    future=model.predict(years.reshape(-1,1))
    return model,pred,years,future,{'R²':r2_score(y,pred),'MAE':mean_absolute_error(y,pred),'RMSE':np.sqrt(mean_squared_error(y,pred)),'Slope/year':model.coef_[0]}

def chart_trend(col,forecast=False,scenario=0):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=a.Year,y=a[col],mode='lines+markers',name='Observed 2021–2025',line=dict(color=colors[col],width=3),marker=dict(size=9)))
    if forecast:
        model,fitted,years,future,_=fit(col)
        fig.add_trace(go.Scatter(x=a.Year,y=fitted,mode='lines',name='Historical fitted trend',line=dict(color='#9ba9bd',dash='dot')))
        if col=='Production':future=future*(1+scenario/100)
        fig.add_trace(go.Scatter(x=years,y=future,mode='lines+markers',name='Projected 2026–2030',line=dict(color='#f7c36d',width=3,dash='dash'),marker=dict(size=8)))
        fig.add_vline(x=2025.5,line_dash='dot',line_color='#667d89')
    fig.update_layout(yaxis_title=units[col],xaxis_title='Year')
    return style(fig)

with st.sidebar:
    st.markdown('## 🌾 PALAY PROJECTION')
    st.caption('AGUSAN DEL SUR · RESEARCH ANALYTICS')
    page=st.radio('NAVIGATION',['Overview','Historical Trends','Growth Analysis','Correlation Explorer','Forecasting Studio','Model Diagnostics','Decision Support','Data & Framework'],label_visibility='collapsed')
    st.divider()
    st.caption('OBSERVED PERIOD · 2021–2025')
    st.caption('PROJECTION PERIOD · 2026–2030')
    st.info('"Data Sources: Philippine Statistics Authority (PSA), PhilRice, and PredictWind | Historical Data: 2021–2025 | Forecast Period: 2026–2030".')

st.markdown(f'<div class="hero"><div class="eyebrow">AGRICULTURAL ANALYTICS  /  {page.upper()}</div><h1>🌾TRENDS IN PALAY PRODUCTION IN AGUSAN DEL SUR (2021-2025): AN ANALYSIS OF PRODUCTION, HARVESTED AREA, YIELD, AND FARMGATE PRICES</h1><div class="muted">Agusan del Sur · Historical monitoring and transparent linear-trend projections</div></div>',unsafe_allow_html=True)

if page=='Overview':
    st.markdown('<div class="section">2025 · KEY PERFORMANCE INDICATORS</div>',unsafe_allow_html=True)
    cols=st.columns(4)
    for i,col in enumerate(metrics):
        now=a[col].iloc[-1];prev=a[col].iloc[-2]
        fmt=f'{now:,.0f}' if i<2 else f'{now:,.4f}' if col=='Farmgate Price' else f'{now:,.2f}'
        cols[i].metric(col,fmt+(' mt' if i==0 else ' ha' if i==1 else ' mt/ha' if i==2 else ' ₱/kg'),f'{(now/prev-1)*100:+.2f}% vs 2024')
    c1,c2=st.columns([1.7,1])
    with c1:st.subheader('Production trajectory');st.plotly_chart(chart_trend('Production',True),use_container_width=True)
    with c2:
        st.subheader('Research snapshot')
        st.metric('Five-year mean production',f'{a.Production.mean():,.0f} mt')
        st.metric('2021–2025 production change',f'{(a.Production.iloc[-1]/a.Production.iloc[0]-1)*100:+.2f}%')
        st.metric('2030 baseline production',f'{fit("Production")[3][-1]:,.0f} mt')
        st.markdown('<div class="insight">Production increased from 2024 to 2025 and reached its highest recorded level in 2025. The 2030 value is an extrapolation, not a guaranteed harvest.</div>',unsafe_allow_html=True)
    st.subheader('Indicator performance')
    cc=st.columns(3)
    for i,col in enumerate(metrics[1:]):
        with cc[i]:st.plotly_chart(chart_trend(col),use_container_width=True)

elif page=='Historical Trends':
    st.subheader('Historical indicator exploration')
    selected=st.selectbox('Choose indicator',metrics)
    st.plotly_chart(chart_trend(selected),use_container_width=True)
    c1,c2,c3,c4=st.columns(4)
    v=a[selected];
    for c,label,value in zip([c1,c2,c3,c4],['Mean','Minimum','Maximum','Sample std. deviation'],[v.mean(),v.min(),v.max(),v.std(ddof=1)]):c.metric(label,f'{value:,.2f}')
    st.dataframe(a[['Year']+metrics].style.format({'Production':'{:,.0f}','Harvested Area':'{:,.0f}','Yield':'{:.2f}','Farmgate Price':'{:.4f}'}),use_container_width=True,hide_index=True)
    st.caption('Standard deviation uses sample ddof=1. Reported yields are not overwritten by production ÷ area. Annual farmgate price for 2025 is the supplied official value ₱14.9325/kg.')

elif page=='Growth Analysis':
    st.subheader('Year-on-year growth rate comparison')
    growth=a[['Year']+metrics].copy();growth[metrics]=growth[metrics].pct_change()*100
    long=growth.dropna().melt('Year',var_name='Indicator',value_name='Growth (%)')
    fig=px.bar(long,x='Year',y='Growth (%)',color='Indicator',barmode='group',color_discrete_map=colors)
    fig.add_hline(y=0,line_color='#9ba9bd')
    st.plotly_chart(style(fig,490),use_container_width=True)
    st.dataframe(growth.style.format({m:'{:+.2f}%' for m in metrics},na_rep='—'),use_container_width=True,hide_index=True)
    st.markdown('<div class="insight">2021 has no growth rate because the dataset does not include 2020. Growth percentages compare changes, not physical quantities.</div>',unsafe_allow_html=True)

elif page=='Correlation Explorer':
    st.subheader('Pearson correlation matrix')
    cols=metrics+['Rainfall','Temperature']
    corr=a[cols].corr(method='pearson')
    fig=px.imshow(corr,text_auto='.3f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r',aspect='auto')
    st.plotly_chart(style(fig,490),use_container_width=True)
    left,right=st.columns([1,2])
    with left:target=st.selectbox('Compare production against',cols[1:])
    with right:
        fig=px.scatter(a,x=target,y='Production',text='Year',color_discrete_sequence=['#6ee0bb'])
        fig.update_traces(marker=dict(size=13),textposition='top center',name='Annual observations',showlegend=True)
        # Fit an ordinary least-squares line for visual interpretation only.
        # This relationship is separate from the year-based forecasting model.
        x_values=a[[target]].to_numpy(dtype=float)
        y_values=a['Production'].to_numpy(dtype=float)
        reg=LinearRegression().fit(x_values,y_values)
        x_line=np.linspace(x_values.min(),x_values.max(),100).reshape(-1,1)
        y_line=reg.predict(x_line)
        fig.add_trace(go.Scatter(x=x_line.ravel(),y=y_line,mode='lines',
            name='OLS regression line',line=dict(color='#f7c36d',width=3)))
        fig.update_layout(xaxis_title=f'{target} ({units[target]})',
                          yaxis_title='Production (metric tons)')
        st.plotly_chart(style(fig),use_container_width=True)
        st.caption(f'Fitted relationship: Production = {reg.intercept_:,.2f} '
                   f'{reg.coef_[0]:+,.4f} × {target}. '
                   'The line illustrates association, not causation or the 2026–2030 forecast.')
    st.metric('Pearson r',f'{a.Production.corr(a[target]):+.4f}')
    st.warning('Exploratory association only (n=5). Correlation does not imply causation; area and yield are mechanically related to production. Semester weather observations are not duplicated as production samples.')

elif page=='Forecasting Studio':
    st.subheader('Independent ordinary least squares projections')
    col=st.selectbox('Forecast indicator',metrics)
    scenario=st.slider('Illustrative production scenario (%)',-10,10,0,1,help='Only changes displayed production projection; not a statistically estimated climate response.') if col=='Production' else 0
    st.plotly_chart(chart_trend(col,True,scenario),use_container_width=True)
    m,p,years,future,score=fit(col)
    if col=='Production':future=future*(1+scenario/100)
    out=pd.DataFrame({'Year':years,'Projected '+col:future})
    st.dataframe(out.style.format({'Projected '+col:'{:,.2f}'}),hide_index=True,use_container_width=True)
    st.download_button('Download projection CSV',out.to_csv(index=False).encode(),'palay_projection.csv','text/csv')
    st.markdown(f'<div class="insight">Model: {col} = {m.intercept_:.4f} + ({m.coef_[0]:.4f} × Year). Each indicator is fitted separately. Future values are baseline trend extrapolations, not observed outcomes.</div>',unsafe_allow_html=True)
    st.caption('Scenario adjustment is illustrative and does not represent a weather-based or confidence-interval forecast.')

elif page=='Model Diagnostics':
    st.subheader('In-sample model evaluation · five annual observations')
    rows=[]
    for col in metrics:
        *_,s=fit(col);rows.append({'Indicator':col,**s})
    df=pd.DataFrame(rows)
    st.dataframe(df.style.format({'R²':'{:.4f}','MAE':'{:,.3f}','RMSE':'{:,.3f}','Slope/year':'+{:,.3f}'}),hide_index=True,use_container_width=True)
    fig=px.bar(df,x='Indicator',y='R²',color='Indicator',color_discrete_map=colors)
    fig.update_layout(showlegend=False);st.plotly_chart(style(fig),use_container_width=True)
    st.warning('R², MAE and RMSE are calculated on the same observations used to fit the model; they are NOT out-of-sample forecast accuracy. Price fit is particularly weak. MAE and RMSE have different units for each indicator.')
    st.caption('All diagnostics are recalculated from the final annual series; minor differences from printed tables may reflect rounding.')

elif page=='Data & Framework':
     st.subheader('Data-Driven Agricultural Forecasting Framework')
     st.caption('Curated annual dataset used by all calculations. Source records were collected and prepared outside this dashboard; the app validates the curated values but does not clean raw PSA, PhilRice, or weather files.')
     st.dataframe(a,use_container_width=True,hide_index=True)
     st.download_button('Download annual dataset',a.to_csv(index=False).encode(),'annual_palay_2021_2025.csv','text/csv')
     st.subheader('Dataset validation checks')
     checks=pd.DataFrame([
         {'Check':'Annual years','Result':'PASS' if a.Year.tolist()==list(range(2021,2026)) else 'FAIL','Detail':'Exactly one observation per year, 2021–2025'},
         {'Check':'Missing values','Result':'PASS' if not a.isna().any().any() else 'FAIL','Detail':'No missing values in the curated annual table'},
         {'Check':'Positive numeric values','Result':'PASS' if (a.drop(columns='Year')>0).all().all() else 'FAIL','Detail':'All six indicator values are positive'},
         {'Check':'2025 official farmgate price','Result':'PASS' if np.isclose(a.loc[a.Year==2025,'Farmgate Price'].iloc[0],14.9325) else 'FAIL','Detail':'Preserved as ₱14.9325/kg, not recomputed from semesters'},
     ])
     st.dataframe(checks,hide_index=True,use_container_width=True)
     st.info('Preprocessing workflow performed before deployment: collect PSA / PhilRice / PredictWind records → standardize year and units → convert numeric fields → check missing and duplicate records → organize annual indicators → validate and load the curated table. The dashboard does not verify source authenticity or reprocess the original raw files.')
     st.subheader('Analysis pipeline')
     steps=[
         ('1 · DATA INPUT', 'Production, Harvested area, Yield, Farmgate Price, Rainfall, Temperature (2021–2025)'),
         ('2 · DATA PREPARATION', 'Collection – Cleaning – Validation – Organization (external preparation; curated table validated in app)'),
         ('3 · STATISTICAL ANALYSIS', 'Descriptive statistics, trend analysis, growth rate, Pearson correlation analysis'),
         ('4 · FORECASTING & EVALUATION', 'Independent linear trend forecasts (2026–2030), in-sample R², MAE, RMSE'),
         ('5 · DECISION SUPPORT OUTPUTS', 'Production outlook, productivity assessment, climate sensitivity interpretation, future supply projection'),
     ]
     for header,description in steps:
         st.markdown(f'<div class="insight"><strong>{header}</strong> — {description}</div>',unsafe_allow_html=True)
     st.subheader('Methodology and formulas')
     st.latex(r'\mathrm{Growth}(\%)=\frac{X_t-X_{t-1}}{X_{t-1}}\times100')
     st.latex(r'r=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2\sum_i(y_i-\bar{y})^2}}')
     st.latex(r'\hat{Y}_t=a+bt')
     st.latex(r'R^2=1-\frac{\sum_i(Y_i-\hat{Y}_i)^2}{\sum_i(Y_i-\bar{Y})^2}')
     st.latex(r'\mathrm{MAE}=\frac{1}{n}\sum_i|Y_i-\hat{Y}_i|')
     st.latex(r'\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i(Y_i-\hat{Y}_i)^2}')
     st.caption('Annual rainfall and temperature are supporting variables for exploratory correlation and climate context; they are not inputs to the univariate year-based forecasts.')

elif page=='Decision Support':
     st.subheader('Decision Support Outputs')
     prod_2030=fit('Production')[3][-1]
     yield_2030=fit('Yield')[3][-1]
     prod_change=(a.Production.iloc[-1]/a.Production.iloc[0]-1)*100
     yield_change=(a.Yield.iloc[-1]/a.Yield.iloc[0]-1)*100
     r_rain=a.Production.corr(a.Rainfall)
     r_temp=a.Production.corr(a.Temperature)
     outputs=pd.DataFrame([
         {'Decision Support Output':'Production Outlook','Analytical Basis':'Historical trends and year-on-year growth','Results and Planning Implications':f'Production changed {prod_change:+.2f}% between 2021 and 2025. Monitor year-to-year fluctuations rather than assuming uninterrupted growth.'},
         {'Decision Support Output':'Productivity Assessment','Analytical Basis':'Yield trend and production identity (Q = A × Y)','Results and Planning Implications':f'Reported yield changed {yield_change:+.2f}% from 2021 to 2025; 2030 yield trend projection is {yield_2030:,.2f} mt/ha. Assess productivity alongside harvested area.'},
         {'Decision Support Output':'Climate Sensitivity Interpretation','Analytical Basis':'Exploratory Pearson correlation with annual weather','Results and Planning Implications':f'Production–rainfall r = {r_rain:+.3f}; production–temperature r = {r_temp:+.3f} (n=5). Associations are not causal effects or forecast inputs.'},
         {'Decision Support Output':'Future Supply Projection','Analytical Basis':'Baseline linear trend forecasting','Results and Planning Implications':f'2030 production baseline = {prod_2030:,.2f} metric tons. Use as an indicative planning scenario, not a guaranteed harvest.'},
     ])
     st.dataframe(outputs,hide_index=True,use_container_width=True,column_config={'Results and Planning Implications':st.column_config.TextColumn(width='large'),'Analytical Basis':st.column_config.TextColumn(width='medium')})
     st.download_button('Download decision support CSV',outputs.to_csv(index=False).encode(),'decision_support_outputs.csv','text/csv')
     st.warning('Decision support interpretations are exploratory because the study uses only five annual observations. The separately forecast production, area, and yield values are not constrained to satisfy Q = A × Y exactly.')

st.divider()
st.caption('PALAY PROJECTION · Dashboard · 2021–2025 observations / 2026–2030 linear-trend extrapolations ·')
