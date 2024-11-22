

#install.packages("mlbench")
#install.packages("torch")

library(mlbench)
library (tidyverse)
library (tidymodels)

tidymodels_prefer ()

library(torch)
install_torch()


# Load and inspect data
data(Sonar)
glimpse (Sonar)

table (Sonar$Class)

summary (Sonar)

# Set seed for reproducibility
set.seed(96)

# Training and test split 
split <- initial_split(Sonar, prop=0.8)

train <- training(split)
test <- testing (split)

# MLP 

# Model definition 
mod <- mlp (
  hidden_units = tune(), 
  epochs = tune(), 
  dropout = tune (),
  learn_rate = tune(),
  activation = "relu",
  engine = "brulee", 
  mode = "classification"
)
# Parameters for tuning
parmSet <- extract_parameter_set_dials (mod)
parmSet

# Recipe 
rec <- recipe (Class ~ ., train) %>%
  step_normalize(all_numeric_predictors())

# Workflow
wf <- workflow () %>% add_model (mod) %>% add_recipe (rec) 

# Tune parameters
tuned <- tune_grid (
  wf,
  resamples = vfold_cv(train, v=5),
  grid = grid_space_filling (parmSet, size = 20), 
  control = control_grid(parallel_over = "everything")
)

# Get the best parameters from the grid search
bestParms <- select_best(tuned, metric='accuracy')

# Update the workflow, fit the model and predict
wfBest <- finalize_workflow(wf, bestParms)

# Fit the model and predict on test dataset
fitted <- fit (wfBest, train)
predicted <- predict (fitted, new_data=test) %>% bind_cols(test) 

# Predicted vs actual metric
predicted %>% metrics(truth = Class , estimate = .pred_class)


# Confusion matrix 
predicted %>% conf_mat(truth = Class , estimate = .pred_class)



