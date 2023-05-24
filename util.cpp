#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#if defined(__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__((__cdecl__))
#endif
#endif

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C"
{

    /*
        @assumptions:
            The blocks must have dimensionality (BATCH_SIZE,)
            The threads must have dimensionality (N,), where
            N * output_thread_slice_size == output_size.

        @param: feature_indices
            A matrix of shape (BATCH_SIZE, max_active_features)
            containing indices of active features for each position
            in a batch. Feature index of -1 means that the slot is empty
            and the weights will not be accumulated for it. Moreover
            no further indices from this block will be considered.
            The indices form an implicit matrix of shape
            (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
            inferred from the memory location (BATCH_SIZE), and the
            second dimension index is stored in the feature_indices matrix.
            The type for feature indices is int32_t.

        @param: feature_values
            A matrix of shape (BATCH_SIZE, max_active_features)
            containing the values (arity) of the corresponding
            feature index in feature_indices.
            The type for the feature value (arity) is float32.

        @param: weight
            The weight matrix of shape (NUM_INPUTS, output_size).
            Weights must be of type float32.

        @param: bias
            The bias vector of shape (output_size,).
            Bias values must be of type float32.

        @param: output
            An output matrix of shape (BATCH_SIZE, output_size).
            It may not be initialized, bias is always copied
            to the output first.
            Output values must have type float32.
    */
    EXPORT void feature_transformer_slice_forward(
        const int32_t *const feature_indices,
        const float *const feature_values,
        const float *const weight,
        const float *const bias,
        float *const output)
    {
        int output_size = 520;
        int output_thread_slice_size = 520;
        int max_active_features = 64;

        float shared_output[520];

        const uint32_t slice_offset = 0;

        float *const output_slice = output + slice_offset;
        const float *const bias_slice = bias + slice_offset;
        float *shared_output_slice = shared_output + slice_offset;

        const int32_t *const feature_index_row = feature_indices;
        const float *const feature_value_row = feature_values;

        for (uint32_t s = 0; s < output_thread_slice_size; ++s)
        {

            shared_output_slice[s] = bias_slice[s];
        }

        for (uint32_t k = 0; k < max_active_features; ++k)
        {

            const int32_t feature_index = feature_index_row[k];
            const float feature_value = feature_value_row[k];
            if (feature_index != -1)
            {

                const float *const weight_slice = weight + feature_index * output_size + slice_offset;
                for (uint32_t s = 0; s < output_thread_slice_size; ++s)
                {

                    shared_output_slice[s] += weight_slice[s] * feature_value;
                }
            }
            else
                break;
        }

        for (uint32_t s = 0; s < output_thread_slice_size; ++s)
        {

            output_slice[s] = shared_output_slice[s];
        }
    }

    /*
        @assumptions:
            The blocks must have dimensionality (BATCH_SIZE,)
            The threads must have dimensionality (N,), where
            N * output_thread_slice_size == output_size.

        @param: feature_indices
            A matrix of shape (BATCH_SIZE, max_active_features)
            containing indices of active features for each position
            in a batch. Feature index of -1 means that the slot is empty
            and the weights will not be accumulated for it. Moreover
            no further indices from this block will be considered.
            The indices form an implicit matrix of shape
            (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
            inferred from the memory location (BATCH_SIZE), and the
            second dimension index is stored in the feature_indices matrix.
            The type for feature indices is int32_t.

        @param: feature_values
            A matrix of shape (BATCH_SIZE, max_active_features)
            containing the values (arity) of the corresponding
            feature index in feature_indices.
            The type for the feature value (arity) is float32.

        @param: weight_grad
            The weight gradient matrix of shape (NUM_INPUTS, output_size).
            The gradient is accumulated, i.e. it must be zero initialized
            on the first call.
            Weights must be of type float32.

        @param: bias_grad
            The bias gradient vector of shape (output_size,).
            The gradient is accumulated, i.e. it must be zero initialized
            on the first call.
            Bias values must be of type float32.

        @param: output_grad
            An output gradient matrix of shape (BATCH_SIZE, output_size).
            Output values must have type float32.
    */
    EXPORT void feature_transformer_slice_backward(
        const int32_t *const feature_indices,
        const float *const feature_values,
        float *const weight_grad,
        float *const bias_grad,
        const float *const output_grad)
    {
        int output_size = 520;
        int output_thread_slice_size = 520;
        int max_active_features = 64;

        float shared_output_grad[520];

        const uint32_t slice_offset = 0;

        const float *const output_grad_slice = output_grad + slice_offset;
        float *const bias_grad_slice = bias_grad + slice_offset;
        float *shared_output_grad_slice = shared_output_grad + slice_offset;

        const int32_t *const feature_index_row = feature_indices;
        const float *const feature_value_row = feature_values;

        for (uint32_t s = 0; s < output_thread_slice_size; ++s)
        {

            shared_output_grad_slice[s] = output_grad_slice[s];
        }

        for (uint32_t s = 0; s < output_thread_slice_size; ++s)
        {

            const float sog = shared_output_grad_slice[s];
            if (sog != 0.0f)
            {
                bias_grad_slice[s] = bias_grad_slice[s] + sog;
            }
        }

        for (uint32_t k = 0; k < max_active_features; ++k)
        {

            const int32_t feature_index = feature_index_row[k];
            const float feature_value = feature_value_row[k];
            if (feature_index != -1)
            {

                float *const weight_grad_slice = weight_grad + feature_index * output_size + slice_offset;
                for (int s = 0; s < output_thread_slice_size; ++s)
                {

                    const float sog = shared_output_grad_slice[s];
                    if (sog != 0.0f)
                    {
                        weight_grad_slice[s] = weight_grad_slice[s] + sog * feature_value;
                    }
                }
            }
            else
                break;
        }
    }
}
