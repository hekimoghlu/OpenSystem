/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
// #define COMMON_RANDOM_FUNCTIONS
#include <CommonCrypto/CommonRandomSPI.h>
#include "ccDispatch.h"
#include <corecrypto/ccaes.h>
#include <corecrypto/ccdrbg.h>
#include <corecrypto/ccrng.h>
#include "ccGlobals.h"
#include "ccErrors.h"
#include "ccdebug.h"

/* These values are ignored, but we have to keep them around for binary compatibility */
static const int ccRandomDefaultStruct;

const CCRandomRef kCCRandomDefault = &ccRandomDefaultStruct;
const CCRandomRef kCCRandomDevRandom = &ccRandomDefaultStruct;

/*
  We don't use /dev/random anymore, use the corecrypto rng instead.
*/
struct ccrng_state *
ccDRBGGetRngState(void)
{
    int status;
    struct ccrng_state *rng = ccrng(&status);
    CC_DEBUG_LOG("ccrng returned %d\n", status);
    return rng;
}

struct ccrng_state *
ccDevRandomGetRngState(void)
{
    return ccDRBGGetRngState();
}

int CCRandomCopyBytes(CCRandomRef rnd, void *bytes, size_t count)
{
    (void) rnd;

    return CCRandomGenerateBytes(bytes, count);
}

CCRNGStatus CCRandomGenerateBytes(void *bytes, size_t count)
{
    int err;
    struct ccrng_state *rng;

    if (0 == count) {
        return kCCSuccess;
    }

    if (NULL == bytes) {
        return kCCParamError;
    }

    rng = ccDRBGGetRngState();
    err = ccrng_generate(rng, count, bytes);
    if (err == CCERR_OK) {
        return kCCSuccess;
    }

    return kCCRNGFailure;
}

CCRNGStatus CCRandomUniform(uint64_t bound, uint64_t *rand)
{
    int err;
    struct ccrng_state *rng;

    rng = ccDRBGGetRngState();
    err = ccrng_uniform(rng, bound, rand);
    if (err == CCERR_OK) {
        return kCCSuccess;
    }

    return kCCRNGFailure;
}
