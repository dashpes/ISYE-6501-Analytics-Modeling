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


library("kknn")

# Test different k values (odd numbers to avoid ties)
k_values <- c(1, 3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51)

cat("=== k-Nearest Neighbors Analysis ===\n")
cat("Testing different k values...\n\n")

# Store results for comparison
knn_results <- data.frame(k = k_values, accuracy = numeric(length(k_values)))

for(j in 1:length(k_values)) {
  k <- k_values[j]
  predictions <- numeric(nrow(data))
  
  cat(sprintf("Testing k = %d...", k))
  
  # Leave-one-out approach: for each data point
  for(i in 1:nrow(data)) {
    # Training set: all data except point i
    train_data <- data[-i, ]
    
    # Test set: just point i (only predictors)
    test_data <- data[i, 1:10, drop = FALSE]  # drop=FALSE keeps it as data frame
    
    # Apply k-nearest neighbors
    knn_model <- kknn(formula = Label ~ ., 
                      train = train_data, 
                      test = test_data,
                      k = k, 
                      scale = TRUE)
    
    # kknn returns fraction of k neighbors that are class 1
    # Convert to binary prediction (>0.5 means predict 1)
    pred_fraction <- knn_model$fitted.values[1]
    predictions[i] <- ifelse(pred_fraction > 0.5, 1, 0)
  }
  
  # Calculate accuracy
  accuracy <- sum(predictions == data$Label) / nrow(data)
  knn_results$accuracy[j] <- accuracy
  
  cat(sprintf(" Accuracy = %.4f\n", accuracy))
}

# Find best k
best_idx <- which.max(knn_results$accuracy)
best_k <- knn_results$k[best_idx]
best_accuracy <- knn_results$accuracy[best_idx]

cat("\n=== RESULTS SUMMARY ===\n")
print(knn_results)

cat(sprintf("\n=== BEST MODEL ===\n"))
cat(sprintf("Best k value: %d\n", best_k))
cat(sprintf("Best accuracy: %.4f (%.2f%%)\n", best_accuracy, best_accuracy * 100))

# Detailed analysis of best k
cat(sprintf("\n=== DETAILED ANALYSIS FOR k = %d ===\n", best_k))

# Run again with best k to get detailed results
predictions <- numeric(nrow(data))
pred_fractions <- numeric(nrow(data))

for(i in 1:nrow(data)) {
  train_data <- data[-i, ]
  test_data <- data[i, 1:10, drop = FALSE]
  
  knn_model <- kknn(formula = Label ~ ., 
                    train = train_data, 
                    test = test_data,
                    k = best_k, 
                    scale = TRUE)
  
  pred_fractions[i] <- knn_model$fitted.values[1]
  predictions[i] <- ifelse(pred_fractions[i] > 0.5, 1, 0)
}

# Prediction distribution
pred_table <- table(predictions)
actual_table <- table(data$Label)

cat("Prediction distribution:\n")
print(pred_table)
cat("Actual distribution:\n")
print(actual_table)

# Confusion matrix
confusion <- table(Predicted = predictions, Actual = data$Label)
cat("\nConfusion Matrix:\n")
print(confusion)

# Additional metrics
true_pos <- confusion[2,2]
true_neg <- confusion[1,1]  
false_pos <- confusion[2,1]
false_neg <- confusion[1,2]

precision <- true_pos / (true_pos + false_pos)
recall <- true_pos / (true_pos + false_neg)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat(sprintf("\nAdditional Metrics for k = %d:\n", best_k))
cat(sprintf("Precision: %.4f\n", precision))
cat(sprintf("Recall: %.4f\n", recall))
cat(sprintf("F1-Score: %.4f\n", f1_score))

# Show some example predictions with their confidence
cat(sprintf("\nSample Predictions (first 10 points):\n"))
cat("Point | Actual | Predicted | Confidence\n")
cat("------|--------|-----------|----------\n")
for(i in 1:10) {
  cat(sprintf("%5d | %6d | %9d | %8.3f\n", i, data$Label[i], predictions[i], pred_fractions[i]))
}