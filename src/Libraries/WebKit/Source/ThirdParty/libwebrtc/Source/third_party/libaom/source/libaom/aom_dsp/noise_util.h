/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef AOM_AOM_DSP_NOISE_UTIL_H_
#define AOM_AOM_DSP_NOISE_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// aom_noise_tx_t is an abstraction of a transform that is used for denoising.
// It is meant to be lightweight and does hold the transformed data (as
// the user should not be manipulating the transformed data directly).
struct aom_noise_tx_t;

// Allocates and returns a aom_noise_tx_t useful for denoising the given
// block_size. The resulting aom_noise_tx_t should be free'd with
// aom_noise_tx_free.
struct aom_noise_tx_t *aom_noise_tx_malloc(int block_size);
void aom_noise_tx_free(struct aom_noise_tx_t *aom_noise_tx);

// Transforms the internal data and holds it in the aom_noise_tx's internal
// buffer. For compatibility with existing SIMD implementations, "data" must
// be 32-byte aligned.
void aom_noise_tx_forward(struct aom_noise_tx_t *aom_noise_tx,
                          const float *data);

// Filters aom_noise_tx's internal data using the provided noise power spectral
// density. The PSD must be at least block_size * block_size and should be
// populated with a constant or via estimates taken from
// aom_noise_tx_add_energy.
void aom_noise_tx_filter(struct aom_noise_tx_t *aom_noise_tx, const float *psd);

// Performs an inverse transform using the internal transform data.
// For compatibility with existing SIMD implementations, "data" must be 32-byte
// aligned.
void aom_noise_tx_inverse(struct aom_noise_tx_t *aom_noise_tx, float *data);

// Aggregates the power of the buffered transform data into the psd buffer.
void aom_noise_tx_add_energy(const struct aom_noise_tx_t *aom_noise_tx,
                             float *psd);

// Returns a default value suitable for denosing a transform of the given
// block_size. The noise "factor" determines the strength of the noise to
// be removed. A value of about 2.5 can be used for moderate denoising,
// where a value of 5.0 can be used for a high level of denoising.
float aom_noise_psd_get_default_value(int block_size, float factor);

// Computes normalized cross correlation of two vectors a and b of length n.
double aom_normalized_cross_correlation(const double *a, const double *b,
                                        int n);

// Validates the correlated noise in the data buffer of size (w, h).
int aom_noise_data_validate(const double *data, int w, int h);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AOM_AOM_DSP_NOISE_UTIL_H_
