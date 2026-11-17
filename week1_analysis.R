# YuvaIntern - Week 1: Data Cleaning and Preliminary Analysis with R
# Dataset: Titanic training data (public CSV)
# Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# 1. Install/load packages (run install.packages only once)
packages <- c("tidyverse", "janitor", "scales")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)
invisible(lapply(packages, library, character.only = TRUE))

# 2. Download and import the public dataset
url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
dir.create("data", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)
download.file(url, "data/titanic.csv", mode = "wb")
titanic_raw <- read.csv("data/titanic.csv", stringsAsFactors = FALSE,
                        na.strings = c("", "NA"))

# 3. Initial inspection
cat("Rows and columns:", dim(titanic_raw), "\n")
str(titanic_raw)
summary(titanic_raw)
missing_before <- sort(colSums(is.na(titanic_raw)), decreasing = TRUE)
print(missing_before)
write.csv(data.frame(column = names(missing_before),
                     missing_count = as.integer(missing_before)),
          "outputs/missing_values_before.csv", row.names = FALSE)

# 4. Clean column names and select useful fields
library(janitor)
titanic <- titanic_raw %>% clean_names() %>%
  select(passenger_id, survived, pclass, name, sex, age, sib_sp,
         parch, ticket, fare, cabin, embarked)

# 5. Missing-value handling
# Age: median imputation is robust to skew/outliers.
age_median <- median(titanic$age, na.rm = TRUE)
titanic$age[is.na(titanic$age)] <- age_median

# Embarked: fill the two missing entries with the most frequent category.
mode_value <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
embarked_mode <- mode_value(titanic$embarked)
titanic$embarked[is.na(titanic$embarked)] <- embarked_mode

# Cabin has a very high proportion of missing entries; remove it from
# the analysis table rather than inventing cabin values.
titanic <- titanic %>% select(-cabin)

# 6. Outlier detection using the 1.5 x IQR rule (flag only; do not delete)
q1 <- quantile(titanic$fare, 0.25, na.rm = TRUE)
q3 <- quantile(titanic$fare, 0.75, na.rm = TRUE)
iqr_fare <- IQR(titanic$fare, na.rm = TRUE)
lower_fare <- q1 - 1.5 * iqr_fare
upper_fare <- q3 + 1.5 * iqr_fare
titanic$fare_outlier <- titanic$fare < lower_fare |
                        titanic$fare > upper_fare
cat("Fare outlier flags:", sum(titanic$fare_outlier, na.rm = TRUE), "\n")

# 7. Transformations and categorical encoding
titanic <- titanic %>%
  mutate(
    sex = factor(sex),
    embarked = factor(embarked),
    pclass = factor(pclass),
    survived = factor(survived, levels = c(0, 1),
                      labels = c("No", "Yes")),
    age_z = as.numeric(scale(age)),
    fare_minmax = (fare - min(fare, na.rm = TRUE)) /
                  (max(fare, na.rm = TRUE) - min(fare, na.rm = TRUE))
  )

# One-hot encode categorical predictors for a modelling-ready example.
encoded <- model.matrix(~ sex + embarked + pclass - 1, data = titanic)
model_ready <- bind_cols(titanic, as.data.frame(encoded))
write.csv(titanic, "outputs/titanic_cleaned.csv", row.names = FALSE)
write.csv(model_ready, "outputs/titanic_model_ready.csv", row.names = FALSE)

# 8. Preliminary analysis
cat("\nCleaned structure:\n")
str(titanic)
cat("\nCleaned summary:\n")
summary(titanic)
cat("\nSurvival counts:\n")
print(table(titanic$survived))
cat("\nSurvival proportion:\n")
print(round(prop.table(table(titanic$survived)) * 100, 1))
cat("\nMean age by survival:\n")
print(titanic %>% group_by(survived) %>%
        summarise(mean_age = mean(age), .groups = "drop"))
cat("\nMean fare by passenger class:\n")
print(titanic %>% group_by(pclass) %>%
        summarise(mean_fare = mean(fare), .groups = "drop"))

# Correlation among numeric variables only
numeric_data <- titanic %>% select(age, sib_sp, parch, fare, age_z, fare_minmax)
cor_matrix <- cor(numeric_data, use = "complete.obs")
print(round(cor_matrix, 2))
write.csv(cor_matrix, "outputs/correlation_matrix.csv")

# 9. Visualisations
p1 <- ggplot(titanic, aes(x = age)) +
  geom_histogram(binwidth = 5, color = "white", fill = "steelblue") +
  labs(title = "Passenger Age Distribution", x = "Age", y = "Count") +
  theme_minimal()
ggsave("outputs/age_distribution.png", p1, width = 7, height = 4, dpi = 150)

p2 <- ggplot(titanic, aes(x = sex, fill = survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Survival Proportion by Sex", x = "Sex",
       y = "Proportion", fill = "Survived") +
  theme_minimal()
ggsave("outputs/survival_by_sex.png", p2, width = 7, height = 4, dpi = 150)

p3 <- ggplot(titanic, aes(x = pclass, y = fare)) +
  geom_boxplot(fill = "tan") +
  labs(title = "Fare Distribution by Passenger Class",
       x = "Passenger class", y = "Fare") +
  theme_minimal()
ggsave("outputs/fare_by_class.png", p3, width = 7, height = 4, dpi = 150)

p4 <- ggplot(titanic, aes(x = age, y = fare, color = survived)) +
  geom_point(alpha = 0.45) +
  labs(title = "Age and Fare by Survival Status",
       x = "Age", y = "Fare", color = "Survived") +
  theme_minimal()
ggsave("outputs/age_fare_scatter.png", p4, width = 7, height = 4, dpi = 150)

# 10. Export reproducibility evidence
capture.output(sessionInfo(), file = "outputs/session_info.txt")
cat("\nAnalysis completed. Check the outputs/ folder for CSVs and charts.\n")
