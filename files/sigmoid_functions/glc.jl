#Generalised Logistic Curve
# A: Upper asymptote
# K: Lower asymptote
# C: Typically takes a value of 1. Otherwise, the upper asymptote is A + (K-A) / (C^(1/v))
# B: Growth rate
# v > 0: affects near which asymptote maximum growth occurs
# Q: is related to the value glc(0)
function glc(x; A=-1.0, K=1.0, C=1.0, B=1.0, v=1.0, Q=1.0)
    return A + ( (K-A) / (C + Q*MathConstants.e^(-B*x) )^(1/v) )
end


# x = [x/10 for x in -50:50]
# y = [glc(i) for i in x ]
# plot(x,y; title="glc" )