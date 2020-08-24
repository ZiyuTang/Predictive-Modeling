library(keras)
library(tensorflow)
library(caret)
library(e1071)
#install_tensorflow(version = 2.0)
Sys.setlocale('LC_ALL','C')
#imdb <- dataset_imdb(num_words = 1000)
#train_data <- imdb$train$x[1:20]
#train_labels <- imdb$train$y
#test_data <- imdb$test$x
#test_labels <- imdb$test$y

# Load data
dt <- read.csv('data_reviews_classified.csv')
dt <- dt[sample(x = 1:nrow(dt), size = 1000), ]

dt$text_id <- NULL
dt$text <- gsub("[[:punct:]]|<.*?>", "", dt$text)
dt$tone <- 1*(dt$tone=="positive")
train_samples <- caret::createDataPartition(c(1:length(dt$tone)),p = 0.9)$Resample1


# Named list mapping words to an integer index.
word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

# Decodes the review. Note that the indices are offset by 3 because 0, 1, and 
# 2 are reserved indices for "padding," "start of sequence," and "unknown."
#decoded_review <- sapply(encode(train_reviews[1]), function(index) {
#  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
#  if (!is.null(word)) word else "?"
#})
#cat(decoded_review)

# Encode the review.
encode <- function(comment) {
  comment = tolower(strsplit(comment, ' ', useBytes = TRUE)[[1]])
  result = sapply(comment, function(word) {
    index <- word_index[[word]] + 3
    if (!is.null(index)) index else 2
  })
  names(result) <- NULL
  result[result <= 3] = 2
  result[result > 10000] = 2
  result[1] <- 1
  unlist(result)
}






train_data <- lapply(dt$text[train_samples], encode)
test_data <- lapply(dt$text[-train_samples], encode)

y_train <- as.numeric(dt$tone[train_samples])
y_test <- as.numeric(dt$tone[-train_samples])

vectorize_sequences <- function(sequences, dimension = 10000) {
  # Creates an all-zero matrix of shape (length(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension) 
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1 
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

val_indices <- 1:100

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

nn <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 8, activation = "relu") %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "sigmoid")

nn %>% compile(
  optimizer = 'rmsprop',
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- nn %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 100,
  batch_size = 10,
  validation_data = list(x_val, y_val)
)

plot(history)


y_hat = (nn %>% predict(x_test)) > 0.5
confusionMatrix(as.factor(as.numeric(1*y_hat)), as.factor(y_test))

nn %>% evaluate(x_test, y_test)

predict_model = function(comment) {
  input = encode(comment) %>% vectorize_sequences() 
  predict(nn, input)
}

nn %>% predict(vectorize_sequences(list(encode('very worth'))))

