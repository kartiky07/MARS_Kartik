def train_model(model, X_train, y_train, epochs=50, batch_size=16):
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size)
    return history
