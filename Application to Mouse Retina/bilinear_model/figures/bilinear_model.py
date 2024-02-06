import numpy as np
import random

random.seed(10)

def create_k_folds(Z, k=5):
    # Get indices of positive and negative samples
    positive_indices = np.argwhere(Z >= 0)
    negative_indices = np.argwhere(Z < 0)

    # Shuffle positive and negative indices separately
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    # Split positive and negative indices into k folds
    positive_folds = np.array_split(positive_indices, k)
    negative_folds = np.array_split(negative_indices, k)

    masks = []
    for i in range(k):
        # Combine positive and negative indices for this fold
        test_indices = np.concatenate((positive_folds[i], negative_folds[i]))

        # Convert to linear indices
        test_indices = np.ravel_multi_index(test_indices.T, Z.shape)

        mask = np.zeros_like(Z, dtype=bool)
        mask[np.unravel_index(test_indices, Z.shape)] = True
        masks.append(mask)

    return masks

def compute_loss(Z, X, A, Y, B, n_factors=1, lambda_a=0.01, lambda_b=0.01, train_mask=None):
    U = np.dot(X, A)
    V = np.dot(Y, B)
    pred = np.dot(U, V.T)
    
    if train_mask is None:
        train_mask = np.ones_like(Z, dtype=bool)
    
    masked_diff = np.zeros_like(Z)
    masked_diff[train_mask] = Z[train_mask] - pred[train_mask]
    
    loss = np.sum(masked_diff**2) + (lambda_a / 2) * np.sum((np.dot(A.T, A))**2) + (lambda_b / 2) * np.sum((np.dot(B.T, B))**2)
    return loss

def transformed_gradient_descent(Z, X, Y, n_factors=1, n_iterations=100000, learning_rate=0.001, lambda_a=0.01, lambda_b=0.01, train_mask=None, verbose=True, tol=1e-6):
    # tol is a parameter for the tolerance for convergence
    n, m = Z.shape
    q = X.shape[1]
    p = Y.shape[1]
    
    # Initialize A and B randomly
    A = np.random.normal(0, 1, (q, n_factors))
    B = np.random.normal(0, 1, (p, n_factors))
    
    if train_mask is None:
        train_mask = np.ones_like(Z, dtype=bool)

    prev_loss = np.inf  # initialize previous loss as infinity

    for iteration in range(n_iterations):
        U = np.dot(X, A)
        V = np.dot(Y, B)
        pred = np.dot(U, V.T)

        masked_Z = np.zeros_like(Z)
        masked_Z[train_mask] = Z[train_mask]

        masked_pred = np.zeros_like(pred)
        masked_pred[train_mask] = pred[train_mask]

        # Update A
        A_grad = np.dot(X.T, np.dot(masked_pred - masked_Z, np.dot(Y, B))) + lambda_a * np.dot(A, (np.dot(A.T, A)))
        A = A - learning_rate * A_grad

        # Update B        
        B_grad = np.dot(Y.T, np.dot((masked_pred - masked_Z).T, np.dot(X, A))) + lambda_b * np.dot(B, (np.dot(B.T, B)))
        B = B - learning_rate * B_grad
        
        if iteration % 100 == 0:
            # Compute loss
            loss = compute_loss(Z, X, A, Y, B, n_factors, lambda_a, lambda_b, train_mask)
            if verbose:
                print(f'Iteration {iteration + 1}, Loss: {loss}')

            # Break loop if convergence
            if np.abs(prev_loss - loss) < tol:
                break

            prev_loss = loss

    return A, B


def cross_validate(Z, X, Y, masks, n_factors=1, n_iterations=100000, learning_rate=0.001, lambda_a=0.01, lambda_b=0.01, verbose = True):
    train_losses = []
    test_losses = []

    for i, test_mask in enumerate(masks):
        print(f'Fold {i + 1}')
        train_mask = ~test_mask
        A, B = transformed_gradient_descent(Z, X, Y, n_factors, n_iterations, learning_rate, lambda_a, lambda_b, train_mask, verbose)
        
        train_loss = compute_loss(Z, X, A, Y, B, n_factors, lambda_a, lambda_b, train_mask)
        test_loss = compute_loss(Z, X, A, Y, B, n_factors, lambda_a, lambda_b, test_mask)
        
        if verbose:
            print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    avg_train_loss = np.mean(train_losses)
    avg_test_loss = np.mean(test_losses)

    if verbose:
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Average Test Loss: {avg_test_loss:.4f}')

    return avg_train_loss, avg_test_loss

def transformed_gradient_descent_output_loss(Z, X, Y, n_factors=1, n_iterations=100000, learning_rate=0.001, lambda_a=0.01, lambda_b=0.01, train_mask=None, verbose=True, tol=1e-6):
    # tol is a parameter for the tolerance for convergence
    n, m = Z.shape
    q = X.shape[1]
    p = Y.shape[1]
    
    # Initialize A and B randomly
    A = np.random.normal(0, 1, (q, n_factors))
    B = np.random.normal(0, 1, (p, n_factors))
    
    if train_mask is None:
        train_mask = np.ones_like(Z, dtype=bool)

    prev_loss = np.inf  # initialize previous loss as infinity
    losses = []

    for iteration in range(n_iterations):
        U = np.dot(X, A)
        V = np.dot(Y, B)
        pred = np.dot(U, V.T)

        masked_Z = np.zeros_like(Z)
        masked_Z[train_mask] = Z[train_mask]

        masked_pred = np.zeros_like(pred)
        masked_pred[train_mask] = pred[train_mask]

        # Update A
        A_grad = np.dot(X.T, np.dot(masked_pred - masked_Z, np.dot(Y, B))) + lambda_a * np.dot(A, (np.dot(A.T, A)))
        A = A - learning_rate * A_grad

        # Update B        
        B_grad = np.dot(Y.T, np.dot((masked_pred - masked_Z).T, np.dot(X, A))) + lambda_b * np.dot(B, (np.dot(B.T, B)))
        B = B - learning_rate * B_grad
        
        # Compute loss
        loss = compute_loss(Z, X, A, Y, B, n_factors, lambda_a, lambda_b, train_mask)
        if verbose:
            print(f'Iteration {iteration + 1}, Loss: {loss}')

        # Break loop if convergence
        if np.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss
        losses.append(loss)

    return A, B, losses