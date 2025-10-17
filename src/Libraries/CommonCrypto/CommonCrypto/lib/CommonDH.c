/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
// #define COMMON_DH_FUNCTIONS
#include <AssertMacros.h>
#include <CommonCrypto/CommonDH.h>
#include <CommonCrypto/CommonRandomSPI.h>
#include "ccDispatch.h"
#include <corecrypto/ccn.h>
#include <corecrypto/cc_priv.h>
#include <corecrypto/ccdh.h>
#include <corecrypto/ccdh_gp.h>
#include "ccErrors.h"
#include "ccGlobals.h"
#include "ccdebug.h"

typedef ccdh_full_ctx_t CCDH;

CCDHRef
CCDHCreate(CCDHParameters dhParameter)
{
    ccdh_const_gp_t gp;
    CCDH retval = NULL;

    CC_DEBUG_LOG("Entering\n");
    if (dhParameter != kCCDHRFC3526Group5) {
        return NULL;
    }

    gp = ccdh_gp_rfc3526group05();
    size_t retSize = ccdh_full_ctx_size(ccdh_ccn_size(gp));
    require((retval = malloc(retSize)) != NULL, error);

    ccdh_ctx_init(gp, ccdh_ctx_public(retval));
    return (CCDHRef)retval;

error:
    free(retval);
    return NULL;
}

void
CCDHRelease(CCDHRef ref)
{
    CC_DEBUG_LOG("Entering\n");
    free((CCDH)ref);
}

int
CCDHGenerateKey(CCDHRef ref, void *output, size_t *outputLength)
{
    CC_DEBUG_LOG("Entering\n");
    CC_NONULLPARM(ref);
    CC_NONULLPARM(output);
    CC_NONULLPARM(outputLength);

    CCDH keyref = (CCDH) ref;

    if (ccdh_generate_key(ccdh_ctx_gp(keyref), ccDRBGGetRngState(), keyref)) {
        return -1;
    }

    size_t size_needed = ccdh_export_pub_size(ccdh_ctx_public(keyref));
    if (size_needed > *outputLength) {
        *outputLength = size_needed;
        return -1;
    }

    *outputLength = size_needed;
    ccdh_export_pub(ccdh_ctx_public(keyref), output);
    return 0;
}


int
CCDHComputeKey(unsigned char *sharedKey, size_t *sharedKeyLen, const void *peerPubKey, size_t peerPubKeyLen, CCDHRef ref)
{
    CC_DEBUG_LOG("Entering\n");
    CC_NONULLPARM(sharedKey);
    CC_NONULLPARM(sharedKeyLen);
    CC_NONULLPARM(peerPubKey);
    CC_NONULLPARM(ref);

    CCDH keyref = (CCDH) ref;
    ccdh_pub_ctx_decl_gp(ccdh_ctx_gp(keyref), peer_pub);

    // Return the expected value in case of error
    size_t size_needed = CC_BITLEN_TO_BYTELEN(ccdh_gp_prime_bitlen(ccdh_ctx_gp(keyref)));
    if (size_needed > *sharedKeyLen) {
        *sharedKeyLen = size_needed;
        return -1;
    }

    // Import key
    if (ccdh_import_pub(ccdh_ctx_gp(keyref), peerPubKeyLen, peerPubKey, peer_pub)) {
        *sharedKeyLen = size_needed;
        return -2;
    }

    // Export secret with no leading zero
    if (ccdh_compute_shared_secret(keyref, peer_pub, sharedKeyLen, sharedKey, ccrng(NULL))) {
        return -3;
    }

    return 0;
}
