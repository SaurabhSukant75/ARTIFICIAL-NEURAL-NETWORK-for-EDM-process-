<h1>Project Title:-Thermo physical modeling of electric discharge  machining process by using Artiﬁcial Neural  Network.</h1>
 
 <h2>1. Introduction</h2> 
 
      1.1 About EDM Process
   <p>The present manufacturing environment is characterized by complexity, interdisciplinary
manufacturing functions and an ever growing demand for new tools and
techniques to solve difficult problems. </p>
   <p>The electrical discharge machining (EDM) process is the most popular among the non-conventional machining
processes. The erosion process of EDM is that the discharge sparks in gap generate enough heat to melt and even
vaporize some of the material on the surface of workpiece, so any difficult-to-cut material can be cut in EDM as
long as the material can conduct electricity.</p>
     <p>for more about EDM-click on link:-https://en.wikipedia.org/wiki/Electrical_discharge_machining#Die-sink_EDM</p>
   <h3>1.2Why ANN Model</h3>
   <p> <b>The complex nature of the process involves simultaneous
interaction of thermal, mechanical, chemical and electrical phenomena, which makes process model very difficult.so,A neural network is used to capture the general relationship between variables of a system that are difficult to relate analytically.<b>
</p>

   <h3>1.3 Advantage of ANN model</h3>
      <p>If we want to find output parameter(MRR,Crater depth) at particular  input  parameter (voltage, current , Ton , Toff ) by <b>Finite Element Method<b>  then we have to do  simulation again & again in ANSYS, which will take:
</p>
        <ol> <li>More time</li>
           <li>Need a  very good  skilled worker for ANSYS SOFTWARE</li> <ol>
     <p>But , we can map input vector([voltage , current, ton, toff ])  to output vector ([MRR,CD])   by using nonlinear statistical data modelling tools like <b>Machine Learning , Deep Learning</b>.we can save </P>   
         <ol><li>Time</li><li>Need of  good skilled worker </li><li>Cost of ANSYS SOFTWARE and skilled worker</li></ol>


</p>
   
   
   
   
   <h4>2. Objective<h4>
   <p> The objective of the Project is to present the application of Artificial Neural Network (ANN) modelling
of the Electrical Discharge Machining (EDM) process. It establishes the best ANN model by comparing
the prediction from different models under the effect of process parameters. In EDM, the motivation is
frequently to get better Material Removal Rate (MRR) with fulfilling better surface quality of machined
components. The vital requirements are as small a radial overcut with minimal tool wear rate. The quality
of a machined surface is very important to fulfilling the growing demands of higher component performance,
durability, and reliability.</p>
   
   
   
   
   
  <h5>3. Data Set<h5>
        <p><b>3.1 Data set discription <b> Data set consist input vector:-         <ol>
         <li>Voltage:-v(volt)</li>
         <li>Current:-I(amp.)</li>
           <li>Dischare On time:ton(sec)</li>
           <li>Dischare Off time:toff(sec)</li>
          </ol>
           Output vector:-<ol>
         <li>Material Removal Rate:-MRR(mm/sec)</li>
         <li>Crater depth:-cd(mm)</li></ol></p>
       <p><b>3.2 Data set question and Problem making :-<b> we have input vector as: [v,I,ton,toff] and output vector:[mrr,cd] and we want to find out output at any arbitrary input.So,it is a clear cut Supervised learning problem and data is continuous so Regression problem.</p>  
         
  <h6> MRR calculator APP</h6> 
  <p>graphic user interface of APP has developed in python. check the GUI_of_APP.jpg </p>
 

