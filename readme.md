   <img src="https://i.pinimg.com/originals/98/65/95/98659526dbbed7d8c4441bfbab9572b0.jpg"
    style="center"
     /> 

# Supply Chain Project

## Pick Data Analysis
#### Project Scope:
Based on publicly available data what is the baseline productivity predicted?
How accurate is the baseline?
Does adding additional features of tenure and shift length improve accuracy of model predictions?

#### Project Inspiration
“We manage what we measure, but frequently we measure what is easy.”

# Unexpected Findings
- Average pick speed did not show much variation year over year.
    - This is unusual because most operations have increasing productivity as an annual goal.
- Average pick speed did not vary with day of week, week of year, or month of year. This may indicate optimized or stagnant processes.
- Total lines is the key feature for this dataset for pick speed prediction.
- Senior operators are doing more complex orders on average, this smooths out the average speed across tenure levels.
    - senior operators should not be penalized for not being faster given how the workload is being divided

# Model Results Summary
Baseline = mean of training dataset = 144 seconds
Baseline Median Absolute Error (MAE) of 95 seconds

Using total lines (order complexity) and if the order was pulled in the last hour of the day
Best Model = 2 degree polynomial features
MAE = 33 seconds

Using only total lines MAE only .01 seconds different
However the explained variance score (EVS) with 2 features was 60% vs 48% with only total lines

Using features of operator tenure and is part time only, expected results to be similar to total lines.
However, results were actually the same MAE as the baseline 95 seconds.

Using 3 features: total lines, operator tenure, is part time
Performance was about the same as the total lines model
MAE = 33 seconds, EVS = 55%