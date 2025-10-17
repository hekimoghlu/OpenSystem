/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#ifndef AOM_AV1_ENCODER_TXB_RDOPT_H_
#define AOM_AV1_ENCODER_TXB_RDOPT_H_

#include "av1/common/blockd.h"
#include "av1/common/txb_common.h"
#include "av1/encoder/encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!\brief Adjust the magnitude of quantized coefficients to achieve better
 * rate-distortion (RD) trade-off.
 *
 * \ingroup coefficient_coding
 *
 * This function goes through each coefficient and greedily choose to lower
 * the coefficient magnitude by 1 or not based on the RD score.
 *
 * The coefficients are processing in reversed scan order.
 *
 * Note that, the end of block position (eob) may change if the original last
 * coefficient is lowered to zero.
 *
 * \param[in]    cpi            Top-level encoder structure
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * \param[in]    tx_size        The transform size
 * \param[in]    tx_type        The transform type
 * \param[in]    txb_ctx        Context info for entropy coding transform block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[out]   rate_cost      The entropy cost of coding the transform block
 * after adjustment of coefficients.
 * \param[in]    sharpness      When sharpness > 0, the function will be less
 * aggressive towards lowering the magnitude of coefficients.
 * In this way, the transform block will contain more high-frequency
 * coefficients and therefore will preserve the sharpness of the reconstructed
 * block.
 */
int av1_optimize_txb(const struct AV1_COMP *cpi, MACROBLOCK *x, int plane,
                     int block, TX_SIZE tx_size, TX_TYPE tx_type,
                     const TXB_CTX *const txb_ctx, int *rate_cost,
                     int sharpness);

/*!\brief Compute the entropy cost of coding coefficients in a transform block.
 *
 * \ingroup coefficient_coding
 *
 * \param[in]    x                    Pointer to structure holding the data for
 the current encoding macroblock.
 * \param[in]    plane                The index of the current plane.
 * \param[in]    block                The index of the current transform block
 in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block.
 * \param[in]    tx_size              The transform size.
 * \param[in]    tx_type              The transform type.
 * \param[in]    txb_ctx              Context info for entropy coding transform
 block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[in]    reduced_tx_set_used  Whether the transform type is chosen from
 * a reduced set.
 */
int av1_cost_coeffs_txb(const MACROBLOCK *x, const int plane, const int block,
                        const TX_SIZE tx_size, const TX_TYPE tx_type,
                        const TXB_CTX *const txb_ctx, int reduced_tx_set_used);

/*!\brief Estimate the entropy cost of coding a transform block using Laplacian
 * distribution.
 *
 * \ingroup coefficient_coding
 *
 * This function compute the entropy costs of the end of block position (eob)
 * and the transform type (tx_type) precisely.
 *
 * Then using av1_cost_coeffs_txb_estimate() (see av1/encoder/txb_rdopt.c) to
 * estimate the entropy costs of coefficients in the transform block.
 *
 * In the end, the function returns the sum of entropy costs of end of block
 * position (eob), transform type (tx_type) and coefficients.
 *
 * Compared to \ref av1_cost_coeffs_txb, this function is much faster but less
 * accurate.
 *
 * \param[in]    x              Pointer to structure holding the data for the
                                current encoding macroblock
 * \param[in]    plane          The index of the current plane
 * \param[in]    block          The index of the current transform block in the
 * macroblock. It's defined by number of 4x4 units that have been coded before
 * the currernt transform block
 * \param[in]    tx_size        The transform size
 * \param[in]    tx_type        The transform type
 * \param[in]    txb_ctx        Context info for entropy coding transform block
 * skip flag (tx_skip) and the sign of DC coefficient (dc_sign).
 * \param[in]    reduced_tx_set_used  Whether the transform type is chosen from
 * a reduced set.
 * \param[in]    adjust_eob     Whether to adjust the end of block position
 (eob)
 * or not.
 * \return       int            Estimated entropy cost of coding the transform
 block.
 */
int av1_cost_coeffs_txb_laplacian(const MACROBLOCK *x, const int plane,
                                  const int block, const TX_SIZE tx_size,
                                  const TX_TYPE tx_type,
                                  const TXB_CTX *const txb_ctx,
                                  const int reduced_tx_set_used,
                                  const int adjust_eob);
#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_ENCODER_TXB_RDOPT_H_
