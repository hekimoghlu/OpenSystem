/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#ifndef _CORECRYPTO_CCDIGEST_INTERNAL_H_
#define _CORECRYPTO_CCDIGEST_INTERNAL_H_

#include <corecrypto/ccdigest_priv.h>

void ccdigest_final_64be(const struct ccdigest_info *di, ccdigest_ctx_t,
    unsigned char *digest);
void ccdigest_final_64le(const struct ccdigest_info *di, ccdigest_ctx_t,
    unsigned char *digest);

CC_INLINE
CC_NONNULL((1))
bool
ccdigest_oid_equal(const struct ccdigest_info *di, ccoid_t oid)
{
	return ccoid_equal(di->oid, oid);
}

#endif /* _CORECRYPTO_CCDIGEST_INTERNAL_H_ */
