/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
#ifndef _CORECRYPTO_CCDIGEST_PRIV_H_
#define _CORECRYPTO_CCDIGEST_PRIV_H_

#include <corecrypto/ccdigest.h>
#include <corecrypto/ccasn1.h>

void ccdigest_final_common(const struct ccdigest_info *di,
                           ccdigest_ctx_t ctx, void *digest);
void ccdigest_final_64be(const struct ccdigest_info *di, ccdigest_ctx_t,
                         unsigned char *digest);
void ccdigest_final_64le(const struct ccdigest_info *di, ccdigest_ctx_t,
                         unsigned char *digest);

CC_INLINE CC_NONNULL_TU((1))
bool ccdigest_oid_equal(const struct ccdigest_info *di, ccoid_t oid) {
    if(di->oid == NULL && CCOID(oid) == NULL) return true;
    if(di->oid == NULL || CCOID(oid) == NULL) return false;
    return ccoid_equal(di->oid, oid);
}

typedef const struct ccdigest_info *(ccdigest_lookup)(ccoid_t oid);

//#include <stdarg.h>

// NOTE(@facekapow):
// i'm not sure entirely sure what Apple was going for with the varargs,
// especially since there was no documentation in the header
//
// my initial assumption was that it was used to try multiple possible OIDs
// and return the digest info for the first one that was found. (un?)fortunately,
// there's no indication of how to determine the length of the varargs
// (NULL could be used as the last element)
//
// therefore, i'm just going to remove the varargs. besides, it's in a private header
// which means that only our own implementation should use it, so we can modify it
// however we want
const struct ccdigest_info *ccdigest_oid_lookup(ccoid_t oid /*, ...*/);

#endif /* _CORECRYPTO_CCDIGEST_PRIV_H_ */
