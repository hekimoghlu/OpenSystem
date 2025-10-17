/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include <stddef.h>
#include <corecrypto/cc.h>
#include <corecrypto/ccdigest.h>
#include <corecrypto/cc_abort.h>

void pdcdigest_fn(const struct ccdigest_info *di, unsigned long len, const void *data, void *digest) {
	ccdigest_di_decl(di, ctx);
	ccdigest_init(di, ctx);
	ccdigest_update(di, ctx, len, data);
	ccdigest_final(di, ctx, (unsigned char *)digest);
	ccdigest_di_clear(di, ctx);
}
