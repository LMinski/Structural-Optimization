F Statistics - It evaluates the overall significance of the model. It compares the full model with an intercept only (no predictors) model. Its value can range between zero and any arbitrary large number. Naturally, higher the F statistics, better the model.

Intercept - This is the βo value. It's the prediction made by model when all the independent variables are set to zero.

coef - This represents regression coefficients for respective variables. It's the value of slope. Let's interpret it for age. We can say, when age is increased by 1 unit, holding other variables constant, charges increase by a value of 261.9106.

Std. Error - This determines the level of variability associated with the estimates. Smaller the standard error of an estimate is, more accurate will be the predictions.

t value - t statistic is generally used to determine variable significance, i.e. if a variable is significantly adding information to the model. t value > 2 suggests the variable is significant. I used it as an optional value as the same information can be extracted from the p value.

p value - It's the probability value of respective variables determining their significance in the model. p value < 0.05 is always desirable.

Durbin Watson Test - This test is used to check autocorrelation. Its value lies between 0 and 4. A DW = 2 value shows no autocorrelation. However, a value between 0 to 2 implies positive autocorrelation, while 2 to 4 implies negative autocorrelation.

Jarque-Bera Test - The Jarque-Bera Test, a type of Lagrange multiplier test, is a test for normality. Normality is one of the assumptions for many statistical tests, like the t test or F test; the Jarque-Bera test is usually run before one of these tests to confirm normality. In general, a large J-B value indicates that errors are not normally distributed.

Skew - A normal distribution has a skew of zero (i.e. it’s perfectly symmetrical around the mean)

Kurtosis - A normal distribution has kurtosis of three, kurtosis tells you how much data is in the tails and gives you an idea about how “peaked” the distribution is.

Akaike Information Criteria (AIC) - You can look at AIC as counterpart of adjusted r square in multiple regression. It's an important indicator of model fit. It follows the rule: Smaller the better. AIC penalizes increasing number of coefficients in the model. In other words, adding more variables to the model wouldn't let AIC increase. It helps to avoid overfitting.