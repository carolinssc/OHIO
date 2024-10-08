tuple Edge{
  int i;
  int j;
}
 
tuple edgeAttrTuple{
    int i;
    int j;
    float g;
    float L;
    float p;
}
 
tuple accTuple{
  int i;
  float n;
}

tuple nodeAttrTuple{
  int i;
  float n;
  float d;
  float cap; 
  float c;
}
 
string path = ...;
{accTuple} desiredInv = ...;
{accTuple} desiredProd = ...;
{int} factory_nodes= ...;
{int} distribution_nodes= ...;
{nodeAttrTuple} nodeAttr = ...;
{edgeAttrTuple} edgeAttr = ...;
{int} f_d_r_nodes = ...;
{accTuple} arrival = ...; 

{Edge} edge = {<i,j>|<i,j, g, L, p> in edgeAttr};
{int} node = {i|<i,n,d,cap, c> in nodeAttr};


float LeadTimes[edge] = [<i,j>:L|<i,j,g,L, p> in edgeAttr]; 
float Holdingcost[edge] = [<i,j>:g|<i,j,g,L, p> in edgeAttr]; 
float price[edge] = [<i,j>:p|<i,j,g,L,p> in edgeAttr];

float demand[node] = [i:d|<i,n,d,cap, c> in nodeAttr];
float inventory[node] = [i:n|<i,n,d,cap, c> in nodeAttr];
float capacity[node] = [i:cap|<i,n,d,cap, c> in nodeAttr];
float prodcapacity[node] = [i:c|<i,n,d,cap, c> in nodeAttr]; 

float desiredInvArray[node] = [i:v|<i,v> in desiredInv]; 
float desiredProdArray[factory_nodes] = [i:v|<i,v> in desiredProd];
float arrivalArray[node]=  [i:v|<i,v> in arrival];

dvar float+ flow[edge];
dvar float error_f[f_d_r_nodes];
dvar float error_p[factory_nodes];
dvar float error_cap[f_d_r_nodes];
dvar float production[factory_nodes];


dexpr float e1 = sum(i in distribution_nodes)abs(error_cap[i]); 
dexpr float e2 = sum(i in factory_nodes)abs(error_p[i]) + sum(i in distribution_nodes)abs(error_f[i]); 

minimize staticLex(e1, e2);  // trying to minimize error first

subject to
{
  forall(i in distribution_nodes)
  {
    sum(e in edge: e.j==i) flow[<e.i, e.j>]  == desiredInvArray[i]+error_f[i];

    sum(e in edge: e.j==i) flow[<e.i, e.j>] + inventory[i] - demand[i] + arrivalArray[i] <= capacity[i]+error_cap[i];
    
  }
  forall(i in factory_nodes)
  { 
    production[i] == desiredProdArray[i] + error_p[i];
    inventory[i] + production[i] - sum(e in edge: e.i==i) flow[<e.i, e.j>] +  arrivalArray[i] <= capacity[i];
    sum(e in edge: e.i==i) flow[<e.i, e.j>] <= inventory[i]+arrivalArray[i];
    production[i] <= prodcapacity[i];

  }
}

main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("flow=[")
  for(var e in thisOplModel.edge)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.flow[e].solutionValue);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.write("error=[")
  for(var i in thisOplModel.distribution_nodes)
       {
         ofile.write("(");
         ofile.write(i);
         ofile.write(",");
         ofile.write(thisOplModel.error_f[i].solutionValue);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.write("production=[")
  for(var i in thisOplModel.factory_nodes)
       {
         ofile.write("(");
         ofile.write(i);
         ofile.write(",");
         ofile.write(thisOplModel.production[i].solutionValue);
         ofile.write(")");
       }
  ofile.writeln("];")

  ofile.close();
}