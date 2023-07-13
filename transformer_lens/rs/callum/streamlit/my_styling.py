import streamlit as st

def styling():
    st.markdown(r"""
<style>
img {
    margin-bottom: 15px;
    max-width: 100%;
}
.myDIV {
    margin-bottom: 15px;
}
.hide {
    display: none;
}
.myDIV:hover + .hide {
    display: block;
    float: left;
    position: absolute;
    z-index: 1;
}
.stAlert h4 {
    padding-top: 0px;
}
.st-ae code {
    padding: 0px !important;
}
label.effi0qh3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 15px;
}
p {
    line-height:1.48em;
}
.st-ae h2 {
    margin-top: -15px;
}
.streamlit-expanderHeader {
    font-size: 1em;
    color: darkblue;
}
.css-ffhzg2 .streamlit-expanderHeader {
    color: lightblue;
}
header {
    background: rgba(255, 255, 255, 0) !important;
}
code:not(pre code) {
    color: red !important;
}
pre code {
    white-space: pre-wrap !important;
    font-size:13px !important;
}
.st-ae code {
    padding: 4px;
}
.css-ffhzg2 .st-ae code: not(stCodeBlock) {
    background-color: black;
}
code:not(h1 code):not(h2 code):not(h3 code):not(h4 code) {
    font-size: 13px;
}
a.contents-el > code {
    color: black;
    background-color: rgb(248, 249, 251);
}
.css-ffhzg2 a.contents-el > code {
    color: orange !important;
    background-color: rgb(26, 28, 36);
}
.css-ffhzg2 code:not(pre code) {
    color: orange !important;
}
.css-ffhzg2 .contents-el {
    color: white !important;
}
.css-fg4pbf blockquote {
    background-color: rgb(231,242,252);
    padding: 15px 20px 5px;
    border-left: 0px solid rgb(230, 234, 241);
}
.katex {
    font-size:18px;
}
h2 .katex, h3 .katex, h4 .katex {
    font-size: unset;
}
ul {
    margin-bottom: 15px;
}
ul.contents {
    line-height:1.3em; 
    list-style:none;
    color-black;
    margin-left: -15px;
}
li.margtop {
    margin-top: 10px !important;
}
ul.contents a, ul.contents a:link, ul.contents a:visited, ul.contents a:active {
    color: black;
    text-decoration: none;
}
ul.contents a:hover {
    color: black;
    text-decoration: underline;
}
.css-ffhzg2 details {
    background-color: #333;
}
.css-fg4pbf details {
    background-color: #eee;
}
details {
    margin-bottom: 10px;
    padding-left: 15px;
    padding-right: 15px;
    padding-top:5px;
    padding-bottom:1px;
    border-radius: 4px;
}
details > div.stCodeBlock {
    margin-bottom: 1rem;
} 
summary {
    margin-bottom: 5px;
}
.css-fg4pbf pre {
    background: rgb(247, 248, 250);
}
.css-fg4pbf code {
    background: rgb(247, 248, 250);
}
.css-fg4pbf code.language-c, .css-fg4pbf pre:has(code.language-c) {
    background-color: #ffe3c4 !important;
}
.css-ffhzg2 code.language-c, .css-ffhzg2 pre:has(code.language-c) {
    background-color: #3d2000 !important;
}
</style>""", unsafe_allow_html=True)