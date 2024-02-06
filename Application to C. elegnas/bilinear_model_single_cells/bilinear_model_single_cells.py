import numpy as np
import random

random.seed(10)

def create_k_folds_symmetric_with_diagonal(Z, W, k=5):
    # Ensure Z is symmetric
    assert np.allclose(Z, Z.T), "Z must be a symmetric matrix"

    # Ensure W is symmetric
    assert np.allclose(W, W.T), "W must be a symmetric matrix"
    
    # Ensure W has the same shape as Z
    assert W.shape == Z.shape, "W must have the same shape as Z"

    # Get indices of the upper triangular part of Z (including the diagonal)
    triu_indices = np.triu_indices_from(Z, k=0)

    # Filter indices where W is non-zero
    filtered_indices = np.column_stack(triu_indices)[W[triu_indices] != 0]

    # Extract the values of Z at the filtered indices
    filtered_Z_values = Z[filtered_indices[:, 0], filtered_indices[:, 1]]

    # Split indices into positive and negative samples based on Z values
    positive_indices = filtered_indices[filtered_Z_values == 1]
    negative_indices = filtered_indices[filtered_Z_values == 0]

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

        # Create mask for both (i, j) and (j, i), including diagonal
        mask = np.zeros_like(Z, dtype=bool)
        mask[test_indices[:, 0], test_indices[:, 1]] = True
        if len(test_indices[0]) == 2:  # Check if off-diagonal elements are present
            mask[test_indices[:, 1], test_indices[:, 0]] = True  # Symmetric part

        masks.append(mask)

    return masks

def compute_loss(Z, X, A, Y, B, W=None, n_factors=1, lambda_a=0.01, lambda_b=0.01, train_mask=None):
    U = np.dot(X, A)
    V = np.dot(Y, B)
    pred = np.dot(U, V.T)
    
    if train_mask is None:
        train_mask = np.ones_like(Z, dtype=bool)
    
    masked_diff = np.zeros_like(Z)
    masked_diff[train_mask] = Z[train_mask] - pred[train_mask]

    if W is not None:
        masked_diff = W*masked_diff
    
    loss = np.sum(masked_diff**2) + (lambda_a / 2) * np.sum(A**2) + (lambda_b / 2) * np.sum(B**2)
    return loss

def transformed_gradient_descent(Z, X, Y, W=None, n_factors=1, n_iterations=100000, learning_rate=0.001, lambda_a=0.01, lambda_b=0.01, train_mask=None, verbose=True, tol=1e-4):
    # tol is a parameter for the tolerance for convergence
    n, m = Z.shape
    q = X.shape[1]
    p = Y.shape[1]
    
    # Initialize A and B randomly
    A = np.random.normal(0, .1, (q, n_factors))
    B = np.random.normal(0, .1, (p, n_factors))
    
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

        if W is not None:
            weighted_diff = W * (masked_pred - masked_Z)
        else:
            weighted_diff = (masked_pred - masked_Z)

        # Update A
        A_grad = 2*np.dot(X.T, np.dot(weighted_diff, np.dot(Y, B))) + lambda_a * A
        A = A - learning_rate * A_grad

        # Update B        
        B_grad = 2*np.dot(Y.T, np.dot((weighted_diff).T, np.dot(X, A))) + lambda_b * B
        B = B - learning_rate * B_grad
        
        if iteration % 100 == 0:
            # Compute loss
            loss = compute_loss(Z, X, A, Y, B, W, n_factors, lambda_a, lambda_b, train_mask)
            if verbose:
                print(f'Iteration {iteration + 1}, Loss: {loss}')

            # Break loop if convergence
            if np.abs(prev_loss - loss) < tol:
                break

            prev_loss = loss

    return A, B


def cross_validate(Z, X, Y, masks, W=None, n_factors=1, n_iterations=100000, learning_rate=0.001, lambda_a=0.01, lambda_b=0.01, verbose = True):
    train_losses = []
    test_losses = []

    for i, test_mask in enumerate(masks):
        print(f'Fold {i + 1}')
        train_mask = ~test_mask
        A, B = transformed_gradient_descent(Z, X, Y, W, n_factors, n_iterations, learning_rate, lambda_a, lambda_b, train_mask, verbose)
        
        train_loss = compute_loss(Z, X, A, Y, B, W, n_factors, lambda_a, lambda_b, train_mask)
        test_loss = compute_loss(Z, X, A, Y, B, W, n_factors, lambda_a, lambda_b, test_mask)
        
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

def transformed_gradient_descent_output_loss(Z, X, Y, W=None, n_factors=1, n_iterations=100000, learning_rate=0.001, lambda_a=0.01, lambda_b=0.01, train_mask=None, verbose=True, tol=1e-6):
    # tol is a parameter for the tolerance for convergence
    n, m = Z.shape
    q = X.shape[1]
    p = Y.shape[1]
    
    # Initialize A and B randomly
    A = np.random.normal(0, .1, (q, n_factors))
    B = np.random.normal(0, .1, (p, n_factors))
    
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

        if W is not None:
            weighted_diff = W * (masked_pred - masked_Z)
        else:
            weighted_diff = (masked_pred - masked_Z)

        # Update A
        A_grad = 2*np.dot(X.T, np.dot(weighted_diff, np.dot(Y, B))) + lambda_a * A
        A = A - learning_rate * A_grad

        # Update B        
        B_grad = 2*np.dot(Y.T, np.dot((weighted_diff).T, np.dot(X, A))) + lambda_b * B
        B = B - learning_rate * B_grad
        
        # Compute loss
        loss = compute_loss(Z, X, A, Y, B, W, n_factors, lambda_a, lambda_b, train_mask)
        if verbose:
            print(f'Iteration {iteration + 1}, Loss: {loss}')

        # Break loop if convergence
        if np.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss
        losses.append(loss)

    return A, B, losses