library("kernlab")
library("kknn")
setwd("~/Documents/GT Masters/ISYE-6501/ISYE-6501-Analytics-Modeling/HW1/")

# Load data
data <- read.delim("../DATA/credit_card_data.txt", header = FALSE)
dataHeader <- read.delim("../DATA/credit_card_data-headers.txt", header = TRUE)

# Optional: assign column names for better readability
colnames(data) <- c("A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","Label")

# Test different C values
c_values <- c(0.01, 0.1, 1, 10, 100, 1000,10000,100000)

for(c in c_values) {
  cat("\n=== C =", c, "===\n")
  
  # Train model
  model <- ksvm(as.matrix(data[,1:10]), as.factor(data[,11]),
                type = "C-svc", kernel = "vanilladot", C = c, scaled = TRUE)
  
  # Calculate coefficients (a1…a10)
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
  cat("Coefficients:\n")
  print(round(a, 6))
  
  # Calculate intercept (a0)
  a0 <- -model@b
  cat("Intercept (a0):", round(a0, 6), "\n")
  
  # Calculate accuracy
  pred <- predict(model, data[,1:10])
  accuracy <- sum(pred == data[,11]) / nrow(data)
  cat("Accuracy:", round(accuracy, 4), "\n")
  
  # Show distribution of predictions (useful for detecting if all predictions are the same)
  pred_table <- table(pred)
  cat("Prediction distribution:\n")
  print(pred_table)
}

# Optional: Store results in a data frame for comparison
results_df <- data.frame(
  C = numeric(),
  Accuracy = numeric(),
  A1 = numeric(), A2 = numeric(), A3 = numeric(), A4 = numeric(), A5 = numeric(),
  A6 = numeric(), A7 = numeric(), A8 = numeric(), A9 = numeric(), A10 = numeric(),
  Intercept = numeric()
)

for(i in 1:length(c_values)) {
  c <- c_values[i]
  model <- ksvm(as.matrix(data[,1:10]), as.factor(data[,11]),
                type = "C-svc", kernel = "vanilladot", C = c, scaled = TRUE)
  
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
  a0 <- -model@b
  pred <- predict(model, data[,1:10])
  accuracy <- sum(pred == data[,11]) / nrow(data)
  
  results_df[i,] <- c(c, accuracy, a, a0)
}

cat("\n=== SUMMARY TABLE ===\n")
print(round(results_df, 4))