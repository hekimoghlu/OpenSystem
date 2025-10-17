/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#ifndef _CORECRYPTO_CC_WORKSPACES_H_
#define _CORECRYPTO_CC_WORKSPACES_H_

CC_PURE size_t sizeof_cc_unit(void);

CC_PURE size_t sizeof_struct_ccbfv_cipher_plain_ctx(void);

CC_PURE size_t sizeof_struct_ccbfv_ciphertext(void);

CC_PURE size_t sizeof_struct_ccbfv_dcrt_plaintext(void);

CC_PURE size_t sizeof_struct_ccbfv_decrypt_ctx(void);

CC_PURE size_t sizeof_struct_ccbfv_encrypt_params(void);

CC_PURE size_t sizeof_struct_ccbfv_galois_key(void);

CC_PURE size_t sizeof_struct_ccbfv_param_ctx(void);

CC_PURE size_t sizeof_struct_ccbfv_plaintext(void);

CC_PURE size_t sizeof_struct_ccbfv_relin_key(void);

CC_PURE size_t sizeof_struct_ccdh_full_ctx(void);

CC_PURE size_t sizeof_struct_ccdh_pub_ctx(void);

CC_PURE size_t sizeof_struct_ccec_full_ctx(void);

CC_PURE size_t sizeof_struct_ccec_pub_ctx(void);

CC_PURE size_t sizeof_struct_ccpolyzp_po2cyc(void);

CC_PURE size_t sizeof_struct_ccpolyzp_po2cyc_base_convert(void);

CC_PURE size_t sizeof_struct_ccpolyzp_po2cyc_block_rng_state(void);

CC_PURE size_t sizeof_struct_ccpolyzp_po2cyc_ctx(void);

CC_PURE size_t sizeof_struct_ccpolyzp_po2cyc_ctx_chain(void);

CC_PURE size_t sizeof_struct_ccrns_mul_modulus(void);

CC_PURE size_t sizeof_struct_ccrsa_full_ctx(void);

CC_PURE size_t sizeof_struct_ccrsa_pub_ctx(void);

CC_PURE size_t sizeof_struct_cczp(void);

CC_PURE cc_size CCBFV_CIPHERTEXT_APPLY_GALOIS_WORKSPACE_N(cc_size degree, cc_size num_ctext_moduli);

CC_PURE cc_size CCBFV_CIPHERTEXT_GALOIS_KEY_SWITCH_WORKSPACE_N(cc_size degree, cc_size num_galois_key_moduli);

CC_PURE cc_size CCBFV_CIPHERTEXT_PLAINTEXT_ADD_WORKSPACE_N(cc_size degree);

CC_PURE cc_size CCBFV_CIPHERTEXT_COEFF_PLAINTEXT_MUL_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_CIPHERTEXT_EVAL_PLAINTEXT_MUL_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_CIPHERTEXT_ROTATE_ROWS_LEFT_WORKSPACE_N(cc_size degree, cc_size num_ctext_moduli);

CC_PURE cc_size CCBFV_CIPHERTEXT_ROTATE_ROWS_RIGHT_WORKSPACE_N(cc_size degree, cc_size num_ctext_moduli);

CC_PURE cc_size CCBFV_CIPHERTEXT_SWAP_COLUMNS_WORKSPACE_N(cc_size degree, cc_size num_ctext_moduli);

CC_PURE cc_size CCBFV_CIPHER_PLAIN_CTX_INIT_WORKSPACE_N(cc_size num_moduli);

CC_PURE cc_size CCBFV_DECODE_SIMD_INT64_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_DECODE_SIMD_UINT64_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_DECRYPT_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_DESERIALIZE_SEEDED_CIPHERTEXT_EVAL_WORKSPACE_N(cc_size degree);

CC_PURE cc_size CCBFV_ENCRYPT_SYMMETRIC_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_ENCRYPT_ZERO_SYMMETRIC_COEFF_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_ENCRYPT_ZERO_SYMMETRIC_EVAL_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_ENCRYPT_ZERO_SYMMETRIC_HELPER_WORKSPACE_N(cc_size degree, cc_size nmoduli);

CC_PURE cc_size CCBFV_GALOIS_KEY_GENERATE_SINGLE_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_GALOIS_KEY_GENERATE_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCBFV_RELIN_KEY_GENERATE_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCDH_POWER_BLINDED_WORKSPACE_N(cc_size n);

CC_PURE cc_size CCEC_AFFINIFY_POINTS_WORKSPACE_N(cc_size n, cc_size npoints);

CC_PURE cc_size CCN_P224_INV_ASM_WORKSPACE_N(cc_size n);

CC_PURE cc_size CCN_P256_INV_ASM_WORKSPACE_N(cc_size n);

CC_PURE cc_size CCN_P384_INV_ASM_WORKSPACE_N(cc_size n);

CC_PURE cc_size CCN_SQR_WORKSPACE_N(cc_size n);

CC_PURE cc_size CCPOLYZP_PO2CYC_BASE_CONVERT_DIVIDE_AND_ROUND_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCPOLYZP_PO2CYC_BASE_CONVERT_INIT_PUNC_PROD_WORKSPACE_N(cc_size num_moduli);

CC_PURE cc_size CCPOLYZP_PO2CYC_BASE_CONVERT_INIT_WORKSPACE_N(cc_size num_moduli);

CC_PURE cc_size CCPOLYZP_PO2CYC_CTX_Q_PROD_WORKSPACE_N(cc_size num_moduli);

CC_PURE cc_size CCPOLYZP_PO2CYC_CTX_WORKSPACE_N(cc_size degree);

CC_PURE cc_size CCPOLYZP_PO2CYC_CTX_INIT_WORKSPACE_N(cc_size n);

CC_PURE cc_size CCPOLYZP_PO2CYC_DESERIALIZE_POLY_WORKSPACE_N(cc_size degree);

CC_PURE cc_size CCPOLYZP_PO2CYC_RANDOM_TERNARY_WORKSPACE_N(cc_size degree);

CC_PURE cc_size CCPOLYZP_PO2CYC_RANDOM_UNIFORM_WORKSPACE_N(cc_size degree);

CC_PURE cc_size CCPOLYZP_PO2CYC_RANDOM_CBD_WORKSPACE_N(cc_size degree);

CC_PURE cc_size CCPOLYZP_PO2CYC_SERIALIZE_POLY_WORKSPACE_N(cc_size degree);

CC_PURE cc_size CCPOLYZP_PO2CYC_WORKSPACE_N(cc_size degree, cc_size num_moduli);

CC_PURE cc_size CCRSA_CRT_POWER_BLINDED_WORKSPACE_N(cc_size n);

#include "cc_workspaces_generated.h"

#endif // _CORECRYPTO_CC_WORKSPACES_H_
