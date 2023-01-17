load census1994
adultdata = rmmissing(adultdata);
adulttest = rmmissing(adulttest);

tic
[Mdl,~,s] = ocsvm(adultdata,GradientTolerance=0.01);
t = toc



% [Mdl, tf] = ocsvm(X, PredictorNames=sX,ContaminationFraction=p1, KernelScale=p2, Lambda=p3, NumExpansionDimensions=p4, ...
%                 StandardizeData=p5, BetaTolerance=p6, ...
%                 GradientTolerance=p7, IterationLimit=p8);