/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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
#ifndef _CORECRYPTO_CCDRBG_DF_H_
#define _CORECRYPTO_CCDRBG_DF_H_

#include <corecrypto/cc.h>
#include <corecrypto/ccmode_impl.h>

// This is an interface for derivation functions for DRBGs to convert
// high-entropy inputs into key material. Because this interface is
// intended for internal usage, we declare only the type names and
// initialization functions here.

typedef struct ccdrbg_df_ctx ccdrbg_df_ctx_t;

struct ccdrbg_df_ctx {
    int (*derive_keys)(const ccdrbg_df_ctx_t *ctx,
                       size_t inputs_count,
                       const cc_iovec_t *inputs,
                       size_t keys_nbytes,
                       void *keys);
};

// This is a block-cipher-based instantiation of the derivation
// function for use in the CTR-DRBG.

typedef struct ccdrbg_df_bc_ctx ccdrbg_df_bc_ctx_t;

struct ccdrbg_df_bc_ctx {
    ccdrbg_df_ctx_t df_ctx;
    const struct ccmode_cbc *cbc_info;
    size_t key_nbytes;

    // See ccmode_impl.h.
    cc_ctx_decl_field(cccbc_ctx, CCCBC_MAX_CTX_SIZE, cbc_ctx);
};

/*!
  @function ccdrbg_df_bc_init
  @abstract Initialize a block-cipher-based derivation function
  @param ctx The derivation function context
  @param cbc_info A descriptor for a CBC mode of a block cipher
  @param key_nbytes The length of the key to use in the derivation function

  @discussion Note that a fixed key is used internally, so only the
  key length needs to be specified.
  @return 0 if successful; negative otherwise
*/
CC_WARN_RESULT
CC_NONNULL_ALL
int ccdrbg_df_bc_init(ccdrbg_df_bc_ctx_t *ctx,
                      const struct ccmode_cbc *cbc_info,
                      size_t key_nbytes);

#endif /* _CORECRYPTO_CCDRBG_DF_H_ */
