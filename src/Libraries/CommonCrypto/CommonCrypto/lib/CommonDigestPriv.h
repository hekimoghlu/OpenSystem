/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
/*
 * CommonDigestPriv.h - private typedefs and defines for ComonCrypto digest routines
 */
 
#ifndef	_COMMON_DIGEST_PRIV_H_
#define _COMMON_DIGEST_PRIV_H_

#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonDigestSPI.h>

// This has to fit in 1032 bytes for static context clients - until we move them.
typedef struct ccDigest_s {
    const struct ccdigest_info *di;
    uint8_t            md[512];
} CCDigestCtx_t, *CCDigestCtxPtr;


// This should remain internal only.  This bridges the CommonCrypto->corecrypto structures

const struct ccdigest_info *
CCDigestGetDigestInfo(CCDigestAlgorithm algorithm);

#endif	/* _COMMON_DIGEST_PRIV_H_ */
