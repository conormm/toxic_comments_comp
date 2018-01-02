library(tidyverse)
library(here)
library(stringi)
library(stringr)

train <- read_csv("data/train.csv")
glimpse(train)

train %>% map_if(is_integer, ~sum(.))
