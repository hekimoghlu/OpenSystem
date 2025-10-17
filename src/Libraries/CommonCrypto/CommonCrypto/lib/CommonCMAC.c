/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
// #define COMMON_CMAC_FUNCTIONS

#define CC_CHANGEFUNCTION_28544056_cccmac_init 1

#include  <CommonCrypto/CommonCMACSPI.h>
#include "CommonCryptorPriv.h"
#include "ccdebug.h"

#include <corecrypto/cccmac.h>
#include <corecrypto/ccaes.h>

void CCAESCmac(const void *key,
               const uint8_t *data,
               size_t dataLength,			/* length of data in bytes */
               void *macOut)				/* MAC written here */
{
    cccmac_one_shot_generate(ccaes_cbc_encrypt_mode(),
                              CCAES_KEY_SIZE_128, key,
                              dataLength, data,
                              CCAES_BLOCK_SIZE, macOut);
}

struct CCCmacContext {
    cccmac_ctx_t ctxptr;
};

CCCmacContextPtr
CCAESCmacCreate(const void *key, size_t keyLength)
{
    // Allocations
    CCCmacContextPtr retval = (CCCmacContextPtr) malloc(sizeof(struct CCCmacContext));
    if(!retval) return NULL;

    const struct ccmode_cbc *cbc = ccaes_cbc_encrypt_mode();
    retval->ctxptr = malloc(cccmac_ctx_size(cbc));
    if(retval->ctxptr == NULL) {
        free(retval);
        return NULL;
    }

    // Initialization (key length check)
    if (key==NULL
        || cccmac_init(cbc, retval->ctxptr,
                    keyLength, key)!=0) {
        free(retval->ctxptr);
        free(retval);
        return NULL;
    }
    
    return retval;
}

void CCAESCmacUpdate(CCCmacContextPtr ctx, const void *data, size_t dataLength) {
    cccmac_update(ctx->ctxptr,dataLength,data);
}

void CCAESCmacFinal(CCCmacContextPtr ctx, void *macOut) {
    cccmac_final_generate(ctx->ctxptr, 16, macOut);
}

void CCAESCmacDestroy(CCCmacContextPtr ctx) {
    if(ctx) {
        free(ctx->ctxptr);
        free(ctx);
    }
}

size_t
CCAESCmacOutputSizeFromContext(CCCmacContextPtr ctx) {
    return cccmac_cbc(ctx->ctxptr)->block_size;
}

