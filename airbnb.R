#load packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidymodels)) install.packages("tidymodels", repos = "http://cran.us.r-project.org")
if(!require(textrecipes)) install.packages("textrecipes", repos = "http://cran.us.r-project.org")
#if(!require(textfeatures)) install.packages("textfeatures", repos = "http://cran.us.r-project.org")
if(!require(tidytext)) install.packages("tidytext", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")
if(!require(ggmap)) install.packages("ggmap", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(magrittr)) install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(heatmaply)) install.packages("heatmaply", repos = "http://cran.us.r-project.org")
#load dataset
dataset=read_csv("listings.csv")
dataset%<>%mutate(price=parse_number(price))
#splitting the data
set.seed(2807)
split=initial_split(dataset, prop = 0.85)
train_set=training(split)
test_set=testing(split)
#visualizations
train_set%>%
  ggplot(aes(longitude, latitude, color=neighbourhood_cleansed))+
  geom_point(size=1)+scale_color_viridis_d()+theme(legend.position = "none") 
train_set%>%ggplot(aes(price))+geom_density()

#log transforming the price
train_set%>%ggplot(aes(log(price+1)))+geom_density()

train_mu=mean(train_set$price)#calculating the mean for graphs

#latitude and longitude vs price
train_set%>%
  group_by(latitude=round(latitude,2),
           longitude=round(longitude,2))%>%
  summarize(price=mean(price))%>%
  ggplot(aes(longitude, latitude, color=price))+
  geom_point()+scale_color_gradient2(high = "yellow", low = "blue", midpoint = train_mu)


#investigating categorical variables
train_set%>%group_by(neighbourhood_cleansed)%>%summarize(price=mean(price-train_mu))%>%
  ggplot(aes(price,reorder(neighbourhood_cleansed,price)))+geom_col()

train_set%>%group_by(room_type)%>%summarize(price=mean(price-train_mu))%>%
  ggplot(aes(price,reorder(room_type,price)))+geom_col()

train_set%>%
  group_by(property_type)%>%
  summarize(price=mean(price))%>%ggplot(aes(reorder(property_type,price), price))+
  geom_col()

#generating folds for modeling
folds=vfold_cv(train_set)

#recipes
#first one uses only location
location_rec=recipe(price~neighbourhood_cleansed+longitude+latitude,data=train_set)%>%
  step_normalize(all_numeric_predictors())%>%
  step_dummy(all_nominal())
#2nd uses property type and room type

type_rec=recipe(price~property_type+room_type,data=train_set)%>%
  step_normalize(all_numeric_predictors())%>%
  step_other(property_type)%>%
  step_dummy(all_nominal())
#3rd one combines the first 2
loc_type_rec=recipe(price~property_type+room_type+neighbourhood_cleansed+longitude+latitude,data=train_set)%>%
  step_normalize(all_numeric_predictors())%>%
  step_other(property_type)%>%
  step_dummy(all_nominal())

#4th one was an attempt as using bathroom information, which consisted of character columns, 
#to generate numeric (number of baths) and categorical (shared baths vs private) predictors
bath_location_rec=recipe(price~neighbourhood_cleansed+property_type+room_type+longitude+latitude+shared_bath+no_bath,data=train_set)%>%
  step_other(property_type)%>%
  step_impute_mode(shared_bath, no_bath)%>%
  step_dummy(all_nominal())%>%step_normalize(all_predictors())

#5th one adds availability_30 to the 3rd recipe

comb_avail_rec=recipe(price~property_type+room_type+neighbourhood_cleansed+longitude+latitude+availability_30,data=train_set)%>%
  step_normalize(all_numeric_predictors())%>%
  step_other(property_type)%>%
  step_dummy(all_nominal())

#6th one adds accomodates to the 5th recipe
accom_rec=recipe(price~property_type+room_type+neighbourhood_cleansed+longitude+latitude
                 +availability_30
                 +accommodates,data=train_set)%>%
  step_normalize(all_numeric_predictors())%>%
  step_other(property_type)%>%
  step_dummy(all_nominal())

#setting tuning options
met=metric_set(rmse)
grid_control=control_grid(extract=extract_model,save_pred = TRUE, save_workflow = TRUE)

#setting up the linear model
lm_model = linear_reg(penalty = tune()) %>% set_engine("glmnet")
#this section was modified and repeated for each recipe to find optimal penalties for each one, the report has the RMSE and penalties for each one.
lm_wflow = 
  workflow() %>% 
  add_model(lm_model) %>% 
  add_recipe(accom_rec)

lm_tune=lm_wflow%>%  tune_grid(resamples=folds,metrics=met, grid=tibble(penalty=seq(0,0.01,0.0001)))
lm_tune%>%collect_metrics()%>%arrange(mean)

autoplot(lm_tune)
#choosing the best linear model
lm_fit=lm_wflow %>%
  finalize_workflow(select_best(lm_tune)) %>%
  fit(train_set)

#setting up xgboost 
bt_model =
  boost_tree(mode="regression",
             mtry = tune(),
             trees = tune(),
             learn_rate = tune()
             #  ,learn_rate = tune(),tree_depth = tune(), min_n = tune(), loss_reduction = tune(), sample_size = tune()
  )%>%
  set_engine("xgboost")

bt_wflow = 
  workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(accom_rec)

#this section was modified multiple times to find the best tuning parameters as doing full grid search was not possible on my computer
bt_tune = bt_wflow %>%
  tune_grid(folds,
            metrics = met,
            control = grid_control,
            grid =crossing(mtry = c(7),
                           trees = seq(1000, 1500, 25), learn_rate=c(0.01)))

autoplot(bt_tune)

bt_tune %>%
  collect_metrics() %>%
  arrange(mean)

#choosing the best xgboost tuning
bt_fit = bt_wflow %>%
  finalize_workflow(select_best(bt_tune)) %>%
  fit(train_set)

#loading the test set and transforming the price to log
test_set=testing(split)
test=test_set%>%mutate(price=log(price+1))

#generating the final predictions using the best linear model
lm_fit%>%augment(test) %>%
  rmse(price, .pred)
#generating the final predictions using the xgboost model
bt_fit%>%augment(test) %>%
  rmse(price, .pred)


