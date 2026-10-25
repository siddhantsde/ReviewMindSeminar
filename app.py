"""ReviewMind — local-first product review intelligence."""
import json
import importlib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_loader import load_sample, load_reviews
from src.preprocessing import clean_reviews
from src.sentiment import add_sentiment
from src.analytics import summary_metrics, product_metrics, key_findings
from src.deep_analysis import data_quality, rating_statistics, rating_distribution, aspect_analysis, monthly_trend, sentiment_by_segment
from src import topic_analysis as _topic_analysis
discover_topics = _topic_analysis.discover_topics
discover_topic_assignments = getattr(_topic_analysis, "discover_topic_assignments", None)
if discover_topic_assignments is None:
    # A running Streamlit process may retain a previously imported module after
    # an in-place upgrade. Reload that module so the expanded topic API is used.
    _topic_analysis = importlib.reload(_topic_analysis)
    discover_topics = _topic_analysis.discover_topics
    discover_topic_assignments = _topic_analysis.discover_topic_assignments
from src.intelligence import pain_points, hidden_problems, trend_analysis, monthly_issue_frequency, rating_disagreements, scenario_estimate, DEFAULT_WEIGHTS, ISSUE_TAXONOMY
from src.recommendations import build_recommendations
from src.llm_service import ollama_status, generate_report
from src.database import save_analysis, list_analyses, get_analysis
from src.report_generator import make_html_report, make_markdown_report

st.set_page_config(page_title="ReviewMind",page_icon="◈",layout="wide")
st.markdown("""<style>
:root{--rm-ink:#172b3a;--rm-muted:#526575;--rm-teal:#087e83;--rm-line:#d9e2e8}
.stApp{background:#f4f7f8;color:var(--rm-ink);font-size:16px}.block-container{padding-top:1.2rem;max-width:1440px;padding-bottom:3rem}
section[data-testid="stSidebar"]{background:#f8fafb!important;border-right:1px solid var(--rm-line);color:#172b3a!important;color-scheme:light}
section[data-testid="stSidebar"] *{color:#172b3a}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] span,section[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{color:#172b3a!important}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] textarea,section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#fff!important;color:#172b3a!important;border-color:#b8c7d1!important}
section[data-testid="stSidebar"] input::placeholder,section[data-testid="stSidebar"] textarea::placeholder{color:#526575!important;opacity:1}
section[data-testid="stSidebar"] button{color:#172b3a!important;border-color:#b8c7d1!important}
section[data-testid="stSidebar"] [data-testid="stFileUploader"]{background:#fff;border:1px solid #d9e2e8;border-radius:10px;padding:8px}
.hero{display:flex;justify-content:space-between;align-items:center;padding:24px 30px;background:#fff;border:1px solid var(--rm-line);border-top:5px solid var(--rm-teal);border-radius:14px;margin-bottom:14px;box-shadow:0 3px 14px #23374708}
.hero h1{margin:0;color:var(--rm-ink);font-size:2rem;letter-spacing:-.03em}.hero p{color:var(--rm-muted);margin:5px 0 0}.hero-mark{font-size:12px;font-weight:700;letter-spacing:.12em;color:var(--rm-teal);text-transform:uppercase}
.kpi{height:100%;background:#fff;border:1px solid var(--rm-line);padding:17px 18px;border-radius:12px;box-shadow:0 2px 8px #23374706}.kpi small{color:var(--rm-muted);font-size:.82rem}.kpi strong{display:block;font-size:1.7rem;color:var(--rm-teal);margin-top:7px;letter-spacing:-.03em}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--rm-line);padding:14px 16px;border-radius:12px}
h1,h2,h3{letter-spacing:-.02em}div[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--rm-line);border-radius:12px;padding:8px}
div[data-testid="stDataFrame"]{border:1px solid var(--rm-line);border-radius:10px;overflow:hidden}
button[kind="primary"]{background:var(--rm-teal);border-color:var(--rm-teal)}
button:focus-visible,input:focus-visible,textarea:focus-visible,[role="combobox"]:focus-visible{outline:3px solid #087e83!important;outline-offset:2px!important}
@media(max-width:760px){.hero{padding:18px;align-items:flex-start}.hero-mark:last-child{display:none}.kpi{padding:13px}.kpi strong{font-size:1.35rem}}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><div><div class="hero-mark">Local review intelligence</div><h1>ReviewMind</h1><p>Turn customer feedback into clear, evidence-backed product decisions.</p></div><div class="hero-mark">ANALYZE · COMPARE · IMPROVE</div></div>',unsafe_allow_html=True)
st.caption("Bundled demonstration reviews are synthetic. Local runs analyze files on this device; hosted deployments process uploads on the host. Ollama generation is optional.")

@st.cache_data(show_spinner=False)
def prepare_reviews(frame):
    cleaned,rejected_rows=clean_reviews(frame)
    return add_sentiment(cleaned),rejected_rows

@st.cache_data(show_spinner=False)
def cached_topics(texts):
    return discover_topics(list(texts))

@st.cache_data(show_spinner=False)
def cached_topic_assignments(texts,n_topics=6,min_topic_size=1):
    return discover_topic_assignments(list(texts),n_topics=n_topics,min_topic_size=min_topic_size)

def reset_to_sample():
    st.session_state.pop("active_raw",None)
    st.session_state.pop("dataset_name",None)
    st.session_state.pop("upload_rejected_rows",None)
    st.session_state.pop("upload_duplicate_count",None)
    st.session_state.pop("upload_row_count",None)
    st.session_state["upload_version"]=st.session_state.get("upload_version",0)+1

st.session_state.setdefault("upload_version",0)
with st.sidebar:
    st.header("ReviewMind")
    upload=st.file_uploader("Upload reviews CSV",type=["csv"],help="Required: product_name, rating, review_text. Ratings may be blank if the source has no star score. Maximum 20 MB.",key=f"review_csv_{st.session_state.upload_version}")
    if upload:
        if upload.size>20*1024*1024: st.error("File exceeds the 20 MB limit."); st.stop()
        try:
            raw=load_reviews(upload); accepted,rejected=clean_reviews(raw)
            st.session_state.upload_row_count=len(raw)
            st.session_state.upload_rejected_rows=rejected
            st.session_state.upload_duplicate_count=int(accepted.is_duplicate.sum())
            st.success(f"Accepted {len(accepted)} · Rejected {len(rejected)}")
            if accepted.empty: st.error("No valid review rows were found."); st.stop()
            exclude=st.checkbox("Exclude duplicate reviews",value=True)
            st.download_button("Download cleaned CSV",accepted.drop(columns=["review_text_clean","is_duplicate"]).to_csv(index=False),"cleaned_reviews.csv","text/csv")
            st.caption("Upload preview")
            st.dataframe(accepted.head(8),use_container_width=True)
            st.session_state.active_raw=accepted.loc[~accepted.is_duplicate] if exclude else accepted
            st.session_state.dataset_name=upload.name
        except Exception as exc:
            st.error(f"Could not load this CSV: {exc}. The sample dataset has not been substituted."); st.stop()
    st.button("Use bundled sample dataset",on_click=reset_to_sample)
    nav_groups={"Overview":["Overview"],"Review insights":["Review Explorer","Sentiment Analysis","Topic Intelligence"],"Issue intelligence":["Customer Pain Points","Issue Priority Matrix","Hidden Problems"],"Trends & comparison":["Trends & Emerging Issues","Product Comparison"],"Reports & history":["AI Product Analyst","Analysis History"],"Data & setup":["Business Impact Explorer","Data Quality","Deep Analytics","Review-to-CSV Prompt"]}
    nav_group=st.radio("Workspace",list(nav_groups),label_visibility="collapsed")
    page=nav_groups[nav_group][0] if len(nav_groups[nav_group])==1 else st.selectbox("Page",nav_groups[nav_group],label_visibility="collapsed")
if "active_raw" in st.session_state: base=st.session_state.active_raw.copy(); dataset_name=st.session_state.dataset_name
else: base=load_sample(); dataset_name="Synthetic sample dataset"
data,rejected=prepare_reviews(base)
if "upload_row_count" not in st.session_state: st.session_state.upload_row_count=len(data)
if data.empty: st.warning("No valid reviews to analyze."); st.stop()
with st.sidebar:
    st.divider(); st.subheader("Active dataset filters")
    cats=sorted(data.category.fillna("Uncategorized").astype(str).unique()); selected_cats=st.multiselect("Category",cats,default=cats)
    products=sorted(data.loc[data.category.isin(selected_cats),"product_name"].unique()); selected_products=st.multiselect("Product",products,default=products)
filtered=data[data.category.isin(selected_cats)&data.product_name.isin(selected_products)].copy(); metrics=summary_metrics(filtered)
def kpis():
    avg=f"{metrics['average_rating']:.2f} / 5" if metrics["average_rating"] is not None else "Not provided"
    vals=[("Reviews analyzed",f"{metrics['review_count']:,}"),("Average rating",avg),("Positive sentiment",f"{metrics['positive_pct']:.1f}%"),("Negative sentiment",f"{metrics['negative_pct']:.1f}%"),("Products",str(metrics['product_count']))]
    for c,(label,value) in zip(st.columns(5),vals): c.markdown(f'<div class="kpi"><small>{label}</small><strong>{value}</strong></div>',unsafe_allow_html=True)
def topics_for(frame): return cached_topics(tuple(frame.review_text.fillna("").astype(str).tolist()))
if page=="Overview":
    kpis(); st.subheader("Overview")
    a,b=st.columns(2)
    with a: st.plotly_chart(px.pie(filtered,names="sentiment",title="Sentiment distribution",color="sentiment",color_discrete_map={"positive":"#2a9d8f","neutral":"#9aa7b2","negative":"#e76f51"}),use_container_width=True)
    with b:
        rated=filtered.dropna(subset=["rating"])
        if len(rated): st.plotly_chart(px.histogram(rated,x="rating",title="Rating distribution",nbins=5,color_discrete_sequence=["#167d87"]),use_container_width=True)
        else: st.info("This dataset has no star ratings. Review text analytics remain available.")
    dated=filtered.dropna(subset=["review_date"]).assign(month=lambda x:x.review_date.dt.to_period("M").astype(str)).groupby("month").size().rename("reviews").reset_index()
    if len(dated): st.plotly_chart(px.line(dated,x="month",y="reviews",title="Review volume over time",markers=True),use_container_width=True)
    pm=product_metrics(filtered); c,d=st.columns(2)
    with c: st.plotly_chart(px.bar(pm,x="product_name",y="average_rating",title="Average rating by product"),use_container_width=True)
    with d: st.plotly_chart(px.bar(pm,x="product_name",y="negative_pct",title="Negative sentiment by product",color="negative_pct",color_continuous_scale="Reds"),use_container_width=True)
    cat=filtered.groupby("category").agg(reviews=("rating","size"),average_rating=("rating","mean"),negative_pct=("sentiment",lambda s:100*(s=="negative").mean())).reset_index()
    st.plotly_chart(px.bar(cat,x="category",y="average_rating",color="category",title="Category comparison · average rating"),use_container_width=True)
    st.subheader("Evidence-backed findings")
    overview_issues=pain_points(filtered,min_reviews=3)
    if len(overview_issues):
        for _,finding in overview_issues.head(3).iterrows():
            st.markdown(f"**{finding.issue}** · {finding.review_count}/{finding.eligible_reviews} eligible reviews ({finding.share_pct}%) · {finding.negative_pct}% negative vs {metrics['negative_pct']:.1f}% overall · {finding.product_count} products.")
            quote=(finding.negative_evidence or finding.positive_evidence)
            if quote: st.caption("Evidence: “"+quote[0]+"”")
            st.caption("Next: "+finding.next_action)
    else: st.info("No recognized issue theme met the minimum evidence threshold. Explore discovered topics and low-rated reviews.")
elif page=="Review Explorer":
    st.subheader("Review Explorer"); c1,c2,c3=st.columns(3)
    ratings=c1.multiselect("Rating",[1,2,3,4,5],default=[1,2,3,4,5]); sentiments=c2.multiselect("Sentiment",["positive","neutral","negative"],default=["positive","neutral","negative"]); verified=c3.selectbox("Verified purchase",["Any","Verified","Unverified"])
    include_unrated=st.checkbox("Include reviews with no star rating",value=True)
    date_values=filtered.review_date.dropna()
    date_range=st.date_input("Review date range",value=(date_values.min().date(),date_values.max().date()) if len(date_values) else (),key="review_date_range")
    query=st.text_input("Search review text"); sort=st.selectbox("Sort by",["Date (newest)","Rating (lowest)","Rating (highest)","Helpful votes"])
    rating_mask=filtered.rating.isin(ratings)
    if include_unrated: rating_mask=rating_mask|filtered.rating.isna()
    result=filtered[rating_mask&filtered.sentiment.isin(sentiments)]
    if verified!="Any": result=result[result.verified_purchase==(verified=="Verified")]
    if isinstance(date_range,(tuple,list)) and len(date_range)==2: result=result[result.review_date.dt.date.between(date_range[0],date_range[1])]
    if query: result=result[result.review_text.str.contains(query,case=False,na=False)]
    sortcol={"Date (newest)":"review_date","Rating (lowest)":"rating","Rating (highest)":"rating","Helpful votes":"helpful_votes"}[sort]; result=result.sort_values(sortcol,ascending=sort not in ["Date (newest)","Rating (highest)","Helpful votes"])
    st.caption(f"{len(result)} reviews match the current filters."); cols=["product_name","rating","sentiment","review_text","review_date","helpful_votes"]
    st.dataframe(result[cols],use_container_width=True,hide_index=True); st.download_button("Download filtered CSV",result.to_csv(index=False),"filtered_reviews.csv","text/csv")
    if len(result):
        chosen=st.selectbox("Inspect a review",result.index,format_func=lambda i:f"{result.loc[i,'product_name']} · {result.loc[i,'rating']} stars · {result.loc[i,'review_text'][:70]}")
        row=result.loc[chosen]; review_text=str(row.review_text); lower=review_text.lower()
        matched=[f"{issue}: {', '.join(term for term in terms if term in lower)}" for issue,terms in ISSUE_TAXONOMY.items() if any(term in lower for term in terms)]
        mixed=any(token in lower for token in [" but "," although "," however "," yet "])
        st.info(f"{review_text}\n\nSentiment: {row.sentiment} · VADER compound score {row.compound:.3f} · rating: {row.rating if pd.notna(row.rating) else 'not provided'}")
        st.write("Matched taxonomy phrases: "+("; ".join(matched) if matched else "No configured issue phrase matched."))
        st.write("Mixed-opinion language cue: "+("possible contrast marker found" if mixed else "no simple contrast marker found"))
        st.caption("This is a grounded text inspection, not a true model explanation. VADER is approximate and may miss sarcasm, context, mixed opinions, and domain-specific meaning.")
elif page=="Sentiment Analysis":
    st.subheader("Sentiment Analysis"); kpis(); counts=filtered.groupby(["rating","sentiment"]).size().reset_index(name="reviews")
    st.plotly_chart(px.bar(counts,x="rating",y="reviews",color="sentiment",barmode="group",title="Sentiment by star rating"),use_container_width=True); st.dataframe(product_metrics(filtered),use_container_width=True,hide_index=True)
    disagreements=rating_disagreements(filtered)
    high=filtered[(filtered.rating>=4)&(filtered.sentiment=="negative")]; low=filtered[(filtered.rating<=2)&(filtered.sentiment=="positive")]
    m1,m2=st.columns(2); m1.metric("Negative text · 4–5 stars",len(high)); m2.metric("Positive text · 1–2 stars",len(low))
    st.caption(f"{len(disagreements)} rating–text disagreements found. VADER is a lexicon-based estimate, not a calibrated probability; sarcasm, context, mixed opinions, and domain language can affect labels.")
    if len(disagreements): st.dataframe(disagreements[["product_name","rating","sentiment","compound","review_text"]],use_container_width=True,hide_index=True)
elif page=="Topic Intelligence":
    st.subheader("Topic Intelligence"); k=st.slider("Topic count",2,12,6); min_size=st.slider("Minimum topic size",1,25,3)
    topics, assignments=cached_topic_assignments(tuple(filtered.review_text.fillna("").astype(str)),n_topics=k,min_topic_size=min_size)
    if topics.empty: st.info("At least five varied review texts are needed; try a lower minimum topic size or upload more varied reviews.")
    else:
        topics["share_pct"]=(100*topics.review_count/max(1,len(filtered))).round(1)
        topic_frames={t:filtered.iloc[assignments[assignments==t].index] for t in topics.topic}
        topics["average_rating"]=[round(float(topic_frames[t].rating.mean()),2) if topic_frames[t].rating.notna().any() else None for t in topics.topic]
        topics["negative_pct"]=[round(100*(topic_frames[t].sentiment=="negative").mean(),1) for t in topics.topic]
        topics["products"]=[", ".join(topic_frames[t].product_name.value_counts().head(3).index.astype(str)) for t in topics.topic]
        st.dataframe(topics,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(topics,x="topic",y="review_count",hover_data=["keywords","share_pct"],title="Review allocation by discovered topic"),use_container_width=True)
        selected_topic=st.selectbox("Inspect topic evidence",topics.topic.tolist())
        # assignments correspond to all non-empty review texts. clean_reviews guarantees text.
        evidence_rows=topic_frames[selected_topic]
        st.dataframe(evidence_rows[["product_name","rating","sentiment","review_text"]],use_container_width=True,hide_index=True)
    st.caption("Topics use deterministic local TF-IDF and NMF. Labels are keyword summaries, not verified categories; taxonomy issues are shown separately.")
elif page=="Customer Pain Points":
    st.subheader("Customer Pain Points")
    st.write("Taxonomy matches are transparent keyword signals; no review is forced into a theme. Percentages use eligible non-empty review text as the denominator.")
    min_n=st.slider("Minimum matching reviews",1,30,3,key="pain_min")
    weights={}
    with st.expander("Configure heuristic priority weights",expanded=False):
        st.caption("Starting weights are assumptions, not validated industry weights. Missing trend or rating components are excluded and remaining weights are renormalized.")
        for key,label in [("frequency","Frequency"),("negative_sentiment","Negative sentiment"),("rating_impact","Rating impact"),("recent_trend","Recent trend"),("product_coverage","Product coverage")]:
            weights[key]=st.slider(label,0,100,int(DEFAULT_WEIGHTS[key]*100),key="pain_weight_"+key)/100
    if not weights or sum(weights.values())<=0: st.error("Set at least one priority weight above zero."); st.stop()
    issues=pain_points(filtered,min_reviews=min_n,weights=weights)
    if issues.empty: st.info(f"No taxonomy theme reached {min_n} reviews in the current filtered dataset. Try reducing the threshold or inspect Topic Intelligence for newly discovered themes.")
    else:
        display=issues.drop(columns=["positive_evidence","negative_evidence","next_action","review_indices","trend"],errors="ignore")
        st.dataframe(display,use_container_width=True,hide_index=True)
        picked=st.selectbox("Investigate an issue",issues.issue.tolist(),key="pain_pick")
        row=issues[issues.issue==picked].iloc[0]
        st.markdown(f"**What is happening?** {row.review_count} of {row.eligible_reviews} eligible reviews ({row.share_pct}%) mention {row.issue}; {row.negative_pct}% of those reviews are negative, compared with {metrics['negative_pct']:.1f}% across the active dataset.")
        st.markdown(f"**Priority:** {row.priority_score}/100 · {row.priority_tier} · configurable heuristic. Components: frequency {row.frequency_component}, negative sentiment {row.negative_component}, rating impact {row.rating_impact_component if pd.notna(row.rating_impact_component) else 'unavailable'}, recent trend {row.trend_component if pd.notna(row.trend_component) else 'unavailable'}, product coverage {row.product_coverage_component}.")
        st.markdown("**Suggested next step:** "+row.next_action)
        if row.trend: st.caption(f"Trend {row.trend['previous_period']} → {row.trend['current_period']}: issue mention share {row.trend['issue_share_pp_change']:+.1f} pp; negative share among mentions {row.trend['negative_pp_change']:+.1f} pp (matching reviews n={row.trend['previous_n']} and {row.trend['current_n']}).")
        ev1,ev2=st.columns(2)
        with ev1:
            st.markdown("**Representative negative evidence**")
            for q in row.negative_evidence: st.markdown("> "+q)
        with ev2:
            st.markdown("**Representative positive evidence**")
            for q in row.positive_evidence: st.markdown("> "+q)
        selected=filtered.loc[filtered.index.isin(row.review_indices)]
        st.markdown("**Reviews behind this finding**"); st.dataframe(selected[["product_name","rating","sentiment","review_text","review_date"]],use_container_width=True,hide_index=True)
elif page=="Issue Priority Matrix":
    st.subheader("Issue Priority Matrix")
    issues=pain_points(filtered,min_reviews=st.slider("Minimum issue reviews",1,30,3,key="matrix_min"))
    if issues.empty: st.info("No recurring issues meet the sample threshold in the active dataset.")
    else:
        fig=px.scatter(issues,x="share_pct",y="negative_pct",size="review_count",color="priority_score",text="issue",hover_data=["priority_tier","priority_score","product_count","eligible_reviews"],color_continuous_scale="YlOrRd",labels={"share_pct":"Share of eligible reviews (%)","negative_pct":"Negative sentiment among issue mentions (%)"},title="Frequency versus negative sentiment")
        fig.update_traces(textposition="top center"); st.plotly_chart(fig,use_container_width=True)
        st.caption("Bubble size reflects matching review count. Color reflects the configurable heuristic score (0–100), not a validated risk probability.")
        st.dataframe(issues.drop(columns=["positive_evidence","negative_evidence","next_action","review_indices","trend"],errors="ignore"),use_container_width=True,hide_index=True)
elif page=="Hidden Problems":
    st.subheader("Hidden Problems · Potential investigation signals")
    st.caption("These transparent rules identify patterns for investigation; negative reviews do not establish that a product is defective.")
    c1,c2,c3=st.columns(3); high_rating=c1.slider("Healthy-looking rating threshold",1.0,5.0,4.0,.1); issue_neg=c2.slider("Minimum issue negative share (%)",0,100,45); min_n=c3.slider("Minimum issue reviews per product",1,30,3)
    issues=pain_points(filtered,min_reviews=min_n); flags=hidden_problems(filtered,issues,high_rating=high_rating,issue_negative_pct=issue_neg,min_issue_reviews=min_n)
    if flags.empty: st.info("No product currently matches the configured high-rating / recurring-negative-theme rule. Small samples may hide issues; see data quality.")
    else:
        st.dataframe(flags.drop(columns="evidence"),use_container_width=True,hide_index=True)
        choice=st.selectbox("Product issue detail",[f"{r.product_name} · {r.issue}" for _,r in flags.iterrows()])
        detail=flags.iloc[[i for i,r in flags.iterrows() if f"{r.product_name} · {r.issue}"==choice][0]]
        st.info(detail.reason)
        for quote in detail.evidence: st.markdown("> "+quote)
        st.dataframe(filtered[(filtered.product_name==detail.product_name)&filtered.index.isin(issues.loc[issues.issue==detail.issue,"review_indices"].iloc[0])][["product_name","rating","sentiment","review_text"]],use_container_width=True,hide_index=True)
    small=filtered.groupby("product_name").size().rename("reviews").reset_index().query("reviews < 10")
    if len(small):
        st.markdown("#### Products with too little evidence for reliable conclusions")
        st.dataframe(small,use_container_width=True,hide_index=True)
        st.caption("These products are surfaced for data-coverage review; no issue conclusion is inferred from a small sample.")
elif page=="Trends & Emerging Issues":
    st.subheader("Trends & Emerging Issues")
    window=st.slider("Comparison window (months)",1,6,1); minimum=st.slider("Minimum reviews per period",2,50,5)
    monthly,changes=trend_analysis(filtered,window_months=window,min_period_reviews=minimum)
    if monthly.empty: st.info("Trend analysis is unavailable because review dates are missing or invalid.")
    else:
        x,y=st.columns(2)
        with x: st.plotly_chart(px.line(monthly,x="month",y="reviews",markers=True,title="Monthly review volume"),use_container_width=True)
        with y: st.plotly_chart(px.line(monthly,x="month",y="negative_pct",markers=True,title="Monthly negative sentiment rate"),use_container_width=True)
        st.plotly_chart(px.line(monthly,x="month",y="average_rating",markers=True,title="Monthly average rating"),use_container_width=True)
        st.dataframe(monthly,use_container_width=True,hide_index=True)
        issue_monthly=monthly_issue_frequency(filtered)
        st.markdown("#### Monthly issue frequency")
        if len(issue_monthly):
            issue_pick=st.multiselect("Issues to display",sorted(issue_monthly.issue.unique()),default=sorted(issue_monthly.issue.unique())[:5])
            selected_monthly=issue_monthly[issue_monthly.issue.isin(issue_pick)]
            st.plotly_chart(px.line(selected_monthly,x="month",y="share_pct",color="issue",markers=True,title="Share of dated reviews mentioning each issue",labels={"share_pct":"Mention share (%)"}),use_container_width=True)
            st.dataframe(selected_monthly,use_container_width=True,hide_index=True)
        else: st.info("No configured issue keywords were found in dated reviews.")
        topic_summary,topic_labels=cached_topic_assignments(tuple(filtered.review_text.fillna("").astype(str)),n_topics=6,min_topic_size=3)
        if len(topic_summary) and len(topic_labels)==len(filtered):
            topic_trends=filtered[["review_date"]].copy(); topic_trends["topic"]=topic_labels.to_numpy(); topic_trends["month"]=topic_trends.review_date.dt.to_period("M").astype(str)
            topic_trends=topic_trends[topic_trends.month!="NaT"].groupby(["month","topic"]).size().rename("reviews").reset_index()
            topic_trends["share_pct"]=topic_trends.groupby("month").reviews.transform(lambda s:100*s/s.sum()).round(1)
            st.markdown("#### Discovered topic mix over time")
            st.plotly_chart(px.line(topic_trends[topic_trends.topic!="Below minimum size"],x="month",y="share_pct",color="topic",markers=True,title="Monthly share of discovered NMF topics"),use_container_width=True)
        else: st.caption("Unsupervised topic shifts are unavailable for the current text volume or vocabulary.")
        if changes.empty: st.info(f"Issue change detection needs at least {minimum} matching reviews in both comparison windows; current data does not meet that threshold.")
        else:
            st.markdown("#### Issue changes · descriptive comparisons")
            st.dataframe(changes,use_container_width=True,hide_index=True)
            up=changes[changes.negative_pp_change>0]
            if len(up):
                top=up.iloc[0]; st.warning(f"{top.issue} negative share increased by {top.negative_pp_change:.1f} percentage points ({top.previous_negative_pct}% → {top.current_negative_pct}%), across {top.previous_reviews} and {top.current_reviews} issue mentions. This is a descriptive change, not a significance claim.")
            inspect=st.selectbox("Inspect issue evidence",changes.issue.tolist(),key="trend_issue_evidence")
            issue_row=pain_points(filtered,min_reviews=1)
            issue_row=issue_row[issue_row.issue==inspect]
            if len(issue_row):
                indices=issue_row.iloc[0].review_indices
                st.dataframe(filtered.loc[filtered.index.isin(indices)][["product_name","rating","sentiment","review_date","review_text"]],use_container_width=True,hide_index=True)
elif page=="Product Comparison":
    st.subheader("Product Comparison"); choices=sorted(filtered.product_name.unique())
    if len(choices)<2: st.info("Choose at least two products using the sidebar filters.")
    else:
        selected_compare=st.multiselect("Products to compare",choices,default=choices[:min(3,len(choices))],key="comparison_products")
        if len(selected_compare)<2: st.info("Select at least two products to compare."); st.stop()
        comparison=product_metrics(filtered[filtered.product_name.isin(selected_compare)]); st.dataframe(comparison,use_container_width=True,hide_index=True)
        if (comparison.review_count<10).any(): st.warning("At least one product has fewer than 10 reviews; comparisons are low-confidence.")
        if comparison.review_count.min()>0 and comparison.review_count.max()/comparison.review_count.min()>=2: st.warning("Largest and smallest review counts differ by at least 2×; interpret rates with sample-size context.")
        if comparison.average_rating.notna().sum()>1: st.caption("Average ratings are descriptive; small differences do not establish a winner. Each product metric includes its review count.")
        st.plotly_chart(px.bar(comparison,x="product_name",y="average_rating",color="product_name",title="Average rating comparison"),use_container_width=True)
        compare_frame=filtered[filtered.product_name.isin(selected_compare)]
        st.dataframe(topics_for(compare_frame),use_container_width=True,hide_index=True)
        product_issues=pain_points(compare_frame,min_reviews=2)
        if len(product_issues):
            st.markdown("#### What distinguishes these products? · issue evidence")
            st.dataframe(product_issues[["issue","review_count","eligible_reviews","share_pct","negative_pct","average_rating","product_count","priority_score"]],use_container_width=True,hide_index=True)
        for product in selected_compare:
            group=filtered[filtered.product_name==product]; stats=rating_statistics(group)
            if stats["ci_low"] is not None: st.caption(f"{product}: approximate 95% mean-rating interval {stats['ci_low']:.2f}–{stats['ci_high']:.2f} (rated n={stats['rated_count']}).")
        disagreement=rating_disagreements(compare_frame)
        st.write(f"Rating–text disagreements: {len(disagreement)} across {len(compare_frame)} selected-product reviews.")
elif page=="Business Impact Explorer":
    st.subheader("Business Impact Explorer")
    st.warning("Illustrative scenario only. Complaints are not unique customers, and an assumed improvement does not establish retention or revenue causality.")
    c1,c2,c3,c4=st.columns(4)
    price=c1.number_input("Assumed average selling price",min_value=0.0,value=100.0,step=10.0)
    units=c2.number_input("Assumed units sold",min_value=0,value=10000,step=100)
    affected=c3.slider("Assumed share affected (%)",0,100,10)
    retained=c4.slider("Assumed retention after improvement (%)",0,100,25)
    try:
        scenario=scenario_estimate(selling_price=price,unit_sales=units,affected_pct=affected,retention_pct=retained)
        a,b,d=st.columns(3); a.metric("Illustrative affected units",f"{scenario['estimated_affected_units']:,.0f}"); b.metric("Illustrative retained units",f"{scenario['illustrative_retained_units']:,.0f}"); d.metric("Illustrative sales value",f"{scenario['illustrative_sales_value']:,.2f}")
        st.caption("Formula: unit sales × assumed affected share × assumed retention share × assumed selling price.")
    except ValueError as exc: st.error(str(exc))
elif page=="Data Quality":
    st.subheader("Dataset Quality & Reliability")
    q=data_quality(filtered); rejected_input=st.session_state.get("upload_rejected_rows",rejected); rejected_count=len(rejected_input)
    q["Rows in loaded dataset"] = st.session_state.upload_row_count
    q["Rows in filtered analysis"] = len(filtered)
    q["Rejected input rows"] = rejected_count
    q["Missing review text"] = int(filtered.review_text.fillna("").astype(str).str.strip().eq("").sum())
    q["Missing product names"] = int(filtered.product_name.fillna("").astype(str).str.strip().eq("").sum())
    if "rating" in rejected_input:
        raw_ratings=rejected_input.rating; numeric_ratings=pd.to_numeric(raw_ratings,errors="coerce")
        supplied=raw_ratings.notna() & raw_ratings.astype(str).str.strip().ne("")
        q["Rows rejected with invalid supplied ratings"] = int((supplied & ~numeric_ratings.between(1,5)).sum())
    else: q["Rows rejected with invalid supplied ratings"] = 0
    q["Rejected rows with missing review text"] = int(rejected_input.review_text.fillna("").astype(str).str.strip().eq("").sum()) if "review_text" in rejected_input else 0
    q["Rejected rows with missing product name"] = int(rejected_input.product_name.fillna("").astype(str).str.strip().eq("").sum()) if "product_name" in rejected_input else 0
    q["Products with <10 reviews"] = int((filtered.groupby("product_name").size()<10).sum())
    q["Missing category"] = int(filtered.category.isna().sum()+(filtered.category.astype(str).str.strip().eq("")).sum())
    q["Missing dates"] = int(filtered.review_date.isna().sum())
    q["Duplicate text rows"] = int(filtered.review_text.fillna("").astype(str).str.lower().duplicated().sum())
    if "upload_duplicate_count" in st.session_state: q["Duplicate rows in upload"] = st.session_state.upload_duplicate_count
    if filtered.review_date.notna().any(): q["Date coverage"] = f"{filtered.review_date.min():%Y-%m-%d} to {filtered.review_date.max():%Y-%m-%d}"
    else: q["Date coverage"] = "Unavailable"
    for label,value in q.items(): st.metric(label,value)
    st.markdown("**Reliability notes**")
    st.write("Small product samples make averages and issue rates unstable. Missing dates disable time comparisons. Missing ratings do not affect text sentiment but are excluded from rating summaries. Exact duplicate text can overweight repeated content; near-duplicate text is not currently detected. Category imbalance can make pooled results reflect the largest category.")
    st.markdown("#### Product and category coverage")
    product_coverage=filtered.groupby("product_name").size().rename("reviews").reset_index().sort_values("reviews")
    category_coverage=filtered.groupby("category",dropna=False).size().rename("reviews").reset_index().sort_values("reviews",ascending=False)
    q1,q2=st.columns(2)
    with q1: st.dataframe(product_coverage,use_container_width=True,hide_index=True)
    with q2: st.dataframe(category_coverage,use_container_width=True,hide_index=True)
    if rejected_count: st.write(f"The upload validation rejected {rejected_count} input rows; correct invalid product names, text, or ratings at source.")
elif page=="Deep Analytics":
    st.subheader("Deep Analytics & Data Quality")
    st.write("Explore rating uncertainty, data completeness, recurring product aspects, customer segments, and review timing. These are descriptive signals from the currently filtered data.")
    quality=data_quality(filtered); quality_items=list(quality.items())
    for start in range(0,len(quality_items),4):
        cols=st.columns(4)
        for col,(label,value) in zip(cols,quality_items[start:start+4]): col.metric(label,value)
    st.caption("Duplicate count reflects the active dataset after the upload duplicate setting. Missing dates and star ratings are not imputed.")
    st.divider(); st.markdown("#### Rating profile")
    stats=rating_statistics(filtered); q1,q2,q3,q4=st.columns(4)
    q1.metric("Rated reviews",stats["rated_count"])
    q2.metric("Mean rating",f"{stats['mean']:.2f} / 5" if stats["mean"] is not None else "Not provided")
    q3.metric("Median rating",f"{stats['median']:.1f} / 5" if stats["median"] is not None else "Not provided")
    q4.metric("Rating spread (SD)",f"{stats['std']:.2f}" if stats["std"] is not None else "Not available")
    if stats["ci_low"] is not None: st.info(f"Approximate 95% confidence interval for the mean rating: **{stats['ci_low']:.2f} to {stats['ci_high']:.2f} / 5** (normal approximation; rated n={stats['rated_count']}). This interval describes sampling uncertainty, not product quality certainty.")
    elif stats["rated_count"]==1: st.info("One rated review is available; a meaningful confidence interval cannot be estimated.")
    dist=rating_distribution(filtered)
    if dist.reviews.sum(): st.plotly_chart(px.bar(dist,x="rating",y="reviews",text="share_pct",title="Rating mix · count with percent share",labels={"rating":"Stars","reviews":"Reviews","share_pct":"Share (%)"},color_discrete_sequence=["#087e83"]),use_container_width=True)
    else: st.info("No ratings are available for this selection. Text and sentiment analysis still include all reviews.")
    st.divider(); st.markdown("#### Review aspects and recurring themes")
    aspects=aspect_analysis(filtered)
    if aspects.empty: st.info("No recognized aspect keywords were found in these reviews.")
    else:
        left,right=st.columns([1.15,1])
        with left: st.plotly_chart(px.bar(aspects.sort_values("negative_pct"),x="negative_pct",y="aspect",orientation="h",color="negative_pct",color_continuous_scale="Reds",hover_data=["mentions","average_rating","evidence_strength"],title="Negative sentiment among reviews mentioning each aspect",labels={"negative_pct":"Negative sentiment (%)","aspect":"Aspect"}),use_container_width=True)
        with right: st.dataframe(aspects,use_container_width=True,hide_index=True)
        st.caption("Aspect mentions use transparent keyword matching. A review can mention several aspects; wording, negation, and sarcasm can affect sentiment. ‘Early signal’ means fewer than 10 mentions; ‘Moderate’ means 10–29; ‘Higher’ means 30 or more.")
    st.divider(); st.markdown("#### Sentiment by customer segment")
    seg=sentiment_by_segment(filtered,"verified_purchase")
    if not seg.empty:
        seg["purchase_status"]=seg.verified_purchase.map({True:"Verified purchase",False:"Not verified"})
        st.plotly_chart(px.bar(seg,x="purchase_status",y=["positive_pct","neutral_pct","negative_pct"],barmode="group",title="Sentiment by verified-purchase status",labels={"purchase_status":"Purchase status","value":"Share of reviews (%)","variable":"Sentiment"}),use_container_width=True)
        st.dataframe(seg.drop(columns="purchase_status"),use_container_width=True,hide_index=True)
    else: st.info("Purchase verification data is not available.")
    st.divider(); st.markdown("#### Review activity over time")
    trend=monthly_trend(filtered)
    if len(trend):
        t1,t2=st.columns(2)
        with t1: st.plotly_chart(px.line(trend,x="month",y="reviews",markers=True,title="Monthly review volume"),use_container_width=True)
        with t2: st.plotly_chart(px.line(trend,x="month",y="negative_pct",markers=True,title="Monthly negative sentiment share",labels={"negative_pct":"Negative share (%)"}),use_container_width=True)
        st.dataframe(trend,use_container_width=True,hide_index=True)
    else: st.info("Add valid review dates to explore trends over time.")
    if filtered.helpful_votes.notna().sum()>2 and filtered.rating.notna().sum()>2:
        paired=filtered[["helpful_votes","rating"]].dropna()
        corr=paired.helpful_votes.rank().corr(paired.rating.rank()) if len(paired)>2 else float("nan")
        st.divider(); st.markdown("#### Helpful votes and ratings")
        if pd.notna(corr): st.write(f"Spearman rank correlation: **{corr:.2f}** across {len(paired)} reviews with both fields. This is an association only and does not imply that helpful votes influence ratings.")
elif page=="AI Product Analyst":
    st.subheader("AI Product Analyst"); status=ollama_status(); st.write(("🟢 " if status["state"]=="available" else "🟠 ")+status["message"])
    st.caption("The optional local Ollama model receives a compact verified evidence package. If unavailable, ReviewMind creates a clearly labeled deterministic rule-based report. Reviews are treated as untrusted quoted content.")
    if st.button("Generate Product Intelligence Report",type="primary"):
        topics=topics_for(filtered); recs=build_recommendations(filtered); issue_df=pain_points(filtered,min_reviews=3); flags=hidden_problems(filtered,issue_df,min_issue_reviews=3); _,changes=trend_analysis(filtered,min_period_reviews=5)
        issue_records=issue_df.drop(columns=["review_indices"],errors="ignore").to_dict("records") if len(issue_df) else []
        hidden_records=flags.to_dict("records") if len(flags) else []
        trend_records=changes.to_dict("records") if len(changes) else []
        facts={"dataset":{"name":dataset_name,"reviews":len(filtered),"product_count":int(filtered.product_name.nunique()),"date_coverage":([str(filtered.review_date.min()),str(filtered.review_date.max())] if filtered.review_date.notna().any() else None)},"metrics":metrics,"product_metrics":product_metrics(filtered).to_dict("records"),"findings":key_findings(filtered),"topics":topics.to_dict("records"),"pain_points":issue_records,"hidden_problems":hidden_records,"trends":trend_records,"recommendations":recs,"data_quality":data_quality(filtered),"limitations":["Synthetic demo reviews are not real consumer research when the bundled sample is active.","VADER labels and keyword taxonomy are approximate.","Priority scores are configurable heuristics, not validated risk models.","Descriptive patterns do not prove causality or statistical significance."],"evidence":{"positive":filtered[filtered.sentiment=="positive"].review_text.head(3).tolist(),"negative":filtered[filtered.sentiment=="negative"].review_text.head(3).tolist(),"pain_points":issue_records,"hidden_problems":hidden_records,"trends":trend_records,"data_quality":data_quality(filtered)}}
        report,method=generate_report(facts); aid=save_analysis(dataset_name,len(filtered),metrics,report,method,details=facts)
        st.session_state.last_report={"report":report,"method":method,"facts":facts,"topics":topics.to_dict("records"),"recommendations":recs,"metrics":metrics,"dataset_name":dataset_name,"analysis_id":aid}
    saved=st.session_state.get("last_report")
    if saved:
        st.markdown(saved["report"]); st.caption("Generation method: "+("Local LLM (Ollama)" if saved["method"]=="local_llm" else "Rule-based evidence report — local LLM unavailable or failed."))
        facts=saved["facts"]
        evidence=facts.get("evidence",{})
        report_html=make_html_report(saved["dataset_name"],saved["metrics"],facts["findings"],saved["topics"],saved["recommendations"],saved["report"],saved["method"],evidence)
        report_md=make_markdown_report(saved["dataset_name"],saved["metrics"],facts["findings"],saved["topics"],saved["recommendations"],saved["report"],saved["method"],evidence)
        st.markdown("#### Export your report")
        exp1,exp2,exp3=st.columns([1,1,1])
        exp1.download_button("Download polished HTML report",report_html,"reviewmind_report.html","text/html",type="primary",use_container_width=True)
        exp2.download_button("Download simple Markdown report",report_md,"reviewmind_report.md","text/markdown",use_container_width=True)
        exp3.download_button("Download metrics CSV",pd.DataFrame([saved["metrics"]]).to_csv(index=False),"reviewmind_metrics.csv","text/csv",use_container_width=True)
    st.subheader("Evidence-based recommendations")
    for r in build_recommendations(filtered):
        with st.container(border=True):
            st.write(r["finding"]); st.write("Suggested action: "+r["action"]); st.caption("Evidence strength: "+r["confidence"])
            for quote in r["evidence"]: st.markdown("> "+quote)
elif page=="Analysis History":
    st.subheader("Analysis History"); history=list_analyses()
    if not history: st.info("Generated reports will be saved here on this device.")
    else:
        st.dataframe(pd.DataFrame(history)[["id","created_at","dataset_name","review_count","generation_method"]],use_container_width=True,hide_index=True)
        aid=st.selectbox("Open saved analysis",[r["id"] for r in history]); record=get_analysis(aid)
        if record:
            st.markdown(record["report"])
            details=json.loads(record.get("details_json","{}")); metrics_saved=json.loads(record["metrics_json"])
            html=make_html_report(record["dataset_name"],metrics_saved,details.get("findings",[]),details.get("topics",[]),details.get("recommendations",[]),record["report"],record["generation_method"],details.get("evidence",{}))
            markdown=make_markdown_report(record["dataset_name"],metrics_saved,details.get("findings",[]),details.get("topics",[]),details.get("recommendations",[]),record["report"],record["generation_method"],details.get("evidence",{}))
            exp1,exp2=st.columns(2)
            exp1.download_button("Download polished HTML report",html,"reviewmind_saved_report.html","text/html",type="primary",use_container_width=True)
            exp2.download_button("Download Markdown report",markdown,"reviewmind_saved_report.md","text/markdown",use_container_width=True)
elif page=="Review-to-CSV Prompt":
    st.subheader("Turn pasted review text into an import-ready CSV")
    st.write("Use this prompt with a language model on your website or in a chat. Paste copied product reviews, text, or a review table after the prompt. It preserves what is present and leaves missing values blank instead of inventing ratings or details.")
    prompt=(__import__("pathlib").Path(__file__).parent/"prompts"/"review_text_to_csv.txt").read_text(encoding="utf-8")
    st.download_button("Download copy-ready prompt",prompt,"reviewmind_review_to_csv_prompt.txt","text/plain",type="primary")
    st.code(prompt,language="text",wrap_lines=True)
    st.info("Blank star ratings are supported: ReviewMind still analyzes the review text and sentiment, while rating-based averages use only rows with an actual rating.")
