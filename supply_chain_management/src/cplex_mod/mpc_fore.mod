tuple Edge{
  int i;
  int j;
}
 
tuple edgeAttrTuple{
    int i;
    int j;
    float g;
    int L;
}
 
tuple accTuple{
  int i;
  float n;
}

tuple nodeAttrTuple{
  int i;
  float c;
  float u;
  float h;
  float s;
  float cap;
  float p; 
  float ud; 
  float b;
  float pr; 
}

tuple accTime{
  int i;
  int t;
  float n;
}

tuple accTimeTuple{
  int i;
  int j;
  int t;
  float n;
}

int T =...;  
string path = ...;
{int} factory_nodes= ...;
{int} distribution_nodes= ...;
{int} retail_nodes = ...;
{nodeAttrTuple} nodeAttr = ...;
{edgeAttrTuple} edgeAttr = ...;
{int} d_r_nodes = ...;
{Edge} reorder_e = ...;
{accTime} demand = ...;
{accTimeTuple} arrival = ...; 


{Edge} edge = {<i,j>|<i,j, g, L> in edgeAttr};
{int} node = {i|<i,c,u,h,s, cap, p, ud, b, pr> in nodeAttr};;
{Edge} reorder_edge = {<i,j>|<i,j> in reorder_e};

int LeadTimes[edge] = [<i,j>:L|<i,j,g,L> in edgeAttr]; 
float pipelineholdingcost[edge] = [<i,j>:g|<i,j,g,L> in edgeAttr]; 
float price[node] = [i:p|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr]; 
float penaltycost[node] = [i:b|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr]; 
float demandInit[node] = [i:ud|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr]; 
float prodInit[node] = [i:pr|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr];

float capacity[node] = [i:c|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr]; 
float capa[node] = [i:cap|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr]; 
float unitoperatingcost[node] = [i:u|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr];
float holdingcost[node] = [i:h|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr];
float stockInit[node] = [i:s|<i,c, u, h, s, cap, p,ud, b, pr> in nodeAttr];

float demandArray[node][1..T] = [i:[t:v]|<i,t,v> in demand];
float arrivalArray[edge][1..T] = [<i,j>:[t:v]|<i,j,t,v> in arrival];

dvar int+ reorder[edge][1..T]; // a j,k,t
dvar float+ production[factory_nodes][0..T]; //p i,t
dvar float+ inventory[node][0..T]; //X,j,t
dvar int+ unfulfilled_demand[d_r_nodes][0..T]; //ut,j,k
dvar int+ realized_demand[d_r_nodes][0..T];
dvar float+ SR[1..T];
dvar float+ PC[1..T];
dvar float+ OC[1..T];
dvar float+ UP[1..T];
dvar float+ HC[1..T];
dvar float+ TC[1..T];
dvar float err[d_r_nodes];

dexpr float e1 = sum(i in d_r_nodes)abs(err[i]); 
dexpr float e2 = -(sum(t in 1..T) (SR[t] - OC[t] - UP[t] - HC[t] - TC[t]));

minimize staticLex(e1, e2);  // trying to minimize error first


subject to 
{ forall(n in node)
    inventory[n][0] == stockInit[n];

  forall(n in d_r_nodes)
  {
    unfulfilled_demand[n][0] == demandInit[n];
    realized_demand[n][0] == 0;
  }
  forall(n in factory_nodes)
    production[n][0] == prodInit[n];

  forall(t in 1..T)
  {
    
    SR[t] == sum(n in d_r_nodes) price[n]*realized_demand[n][t]; 
    OC[t] == sum(n in factory_nodes) unitoperatingcost[n]*production[n][t];
    TC[t] == sum(e in reorder_edge) (pipelineholdingcost[e]*reorder[e][t]*LeadTimes[e]);
    UP[t] == sum(n in d_r_nodes) penaltycost[n]*unfulfilled_demand[n][t];
    HC[t] == sum(n in d_r_nodes) holdingcost[n]*(inventory[n][t] - realized_demand[n][t]) + sum(n in factory_nodes) holdingcost[n]*(inventory[n][t]);

    forall(n in factory_nodes)
    {
      production[n][t] <= capacity[n];
    }
    forall(n in d_r_nodes)
    {
      inventory[n][t] == inventory[n][t-1] -realized_demand[n][t-1] +  sum(e in edge: e.j==n && t - LeadTimes[e] >= 1)reorder[e][t - LeadTimes[e]] + sum(e in edge: e.j==n) arrivalArray[e][t];
      inventory[n][t] - realized_demand[n][t] <= capa[n]+err[n];

      realized_demand[n][t] <= demandArray[n][t] + unfulfilled_demand[n][t-1];
      realized_demand[n][t] <= inventory[n][t]; 
      unfulfilled_demand[n][t] == demandArray[n][t] + unfulfilled_demand[n][t-1] - realized_demand[n][t];
    }

    forall(i in factory_nodes)       
    {
      
      inventory[i][t] == inventory[i][t-1] + production[i][t-1]  - sum(e in edge: e.i==i)reorder[e][t];
      sum(e in edge: e.i==i)reorder[e][t] <= inventory[i][t-1] + production[i][t-1];
      inventory[i][t] <= capa[i];
      
    }
  }
} 

main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
ofile.write("flow=[");
    for (var e in thisOplModel.edge) {
      ofile.write("(");
      ofile.write(e.i);
      ofile.write(",");
      ofile.write(e.j);
      ofile.write(",");
      ofile.write(thisOplModel.reorder[e][1].solutionValue);
      ofile.write(")");
  }
  ofile.writeln("];");
  ofile.write("production=[");
    for (var n in thisOplModel.factory_nodes) {
      ofile.write("(");
      ofile.write(n);
      ofile.write(",");
      ofile.write(thisOplModel.production[n][1].solutionValue);
      ofile.write(")");
  }
    ofile.writeln("];");
  ofile.write("SR=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    ofile.write("(");
      ofile.write(t);
      ofile.write(",");
    ofile.write(thisOplModel.SR[t].solutionValue);
    ofile.write(")");
  }
    ofile.writeln("];");
  ofile.write("PC=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    ofile.write("(");
      ofile.write(t);
      ofile.write(",");
    ofile.write(thisOplModel.PC[t].solutionValue);
    ofile.write(")");
  }
    ofile.writeln("];");
ofile.write("OC=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    ofile.write("(");
      ofile.write(t);
      ofile.write(",");
    ofile.write(thisOplModel.OC[t].solutionValue);
    ofile.write(")");
  }
    ofile.writeln("];");
  ofile.write("UP=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    ofile.write("(");
      ofile.write(t);
      ofile.write(",");
    ofile.write(thisOplModel.UP[t].solutionValue);
    ofile.write(")");
  }
    ofile.writeln("];");
  ofile.write("HC=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
   ofile.write("(");
      ofile.write(t);
      ofile.write(",");
    ofile.write(thisOplModel.HC[t].solutionValue);
    ofile.write(")");
  }
    ofile.writeln("];");
  ofile.write("TC=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    ofile.write("(");
      ofile.write(t);
      ofile.write(",");
    ofile.write(thisOplModel.TC[t].solutionValue);
    ofile.write(")");
  }
  ofile.writeln("];");
  ofile.write("unfulfilled_demand=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    for (var n in thisOplModel.d_r_nodes) {
      ofile.write("(");
      ofile.write(n);
      ofile.write(",");
      ofile.write(t);
      ofile.write(",");
      ofile.write(thisOplModel.unfulfilled_demand[n][t].solutionValue);
      ofile.write(")");
    }
  }
  ofile.writeln("];");
  ofile.write("realized_demand=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    for (var n in thisOplModel.d_r_nodes) {
      ofile.write("(");
      ofile.write(n);
      ofile.write(",");
      ofile.write(t);
      ofile.write(",");
      ofile.write(thisOplModel.realized_demand[n][t].solutionValue);
      ofile.write(")");
    }
  }
  ofile.writeln("];");
  ofile.write("flows=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    for (var e in thisOplModel.edge) {
      ofile.write("(");
      ofile.write(e.i);
      ofile.write(",");
      ofile.write(e.j);
      ofile.write(",");
      ofile.write(t);
      ofile.write(",");
      ofile.write(thisOplModel.reorder[e][t].solutionValue);
      ofile.write(")");
    }
  }
  ofile.writeln("];");
  ofile.write("productions=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    for (var n in thisOplModel.factory_nodes) {
      ofile.write("(");
      ofile.write(n);
      ofile.write(",");
      ofile.write(t);
      ofile.write(",");
      ofile.write(thisOplModel.production[n][t].solutionValue);
      ofile.write(")");
    }
  }
  ofile.writeln("];");
  ofile.write("inventory=[");
  for (var t = 1; t <= thisOplModel.T; t++) {
    for (var n in thisOplModel.node) {
      ofile.write("(");
      ofile.write(n);
      ofile.write(",");
      ofile.write(t);
      ofile.write(",");
      ofile.write(thisOplModel.inventory[n][t].solutionValue);
      ofile.write(")");
    }
  }
  ofile.writeln("];");
  ofile.close();
}

