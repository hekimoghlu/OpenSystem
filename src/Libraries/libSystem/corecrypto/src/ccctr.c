/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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

#include <corecrypto/ccmode_factory.h>
#include <corecrypto/ccstubs.h>

int ccmode_ctr_init(const struct ccmode_ctr* ctr, ccctr_ctx* _ctx, size_t rawkey_len, const void* rawkey, const void* iv) {
	CC_STUB_ERR();
};

int ccmode_ctr_crypt(ccctr_ctx* ctx, size_t nbytes, const void* in, void* out) {
	CC_STUB_ERR();
};

int ccmode_ctr_setctr(const struct ccmode_ctr* ctr, ccctr_ctx* ctx, const void* iv) {
	CC_STUB_ERR();
};
