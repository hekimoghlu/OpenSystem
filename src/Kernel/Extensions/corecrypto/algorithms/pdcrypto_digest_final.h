/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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

//
//  pd_crypto_digest_final.h
//  pdcrypto
//
//  Created by rafirafi on 3/17/16.
//  Copyright (c) 2016 rafirafi. All rights reserved.
//

#ifndef __pdcrypto__pd_crypto_digest_final__
#define __pdcrypto__pd_crypto_digest_final__

#include <corecrypto/ccdigest.h>

__BEGIN_DECLS
void pdcdigest_final_64le(const struct ccdigest_info *di, ccdigest_ctx_t ctx, unsigned char *digest);

void pdcdigest_final_64be(const struct ccdigest_info *di, ccdigest_ctx_t ctx, unsigned char *digest);
__END_DECLS

#endif /* defined(__pdcrypto__pd_crypto_digest_final__) */
