/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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

int ccmode_xts_init(const struct ccmode_xts* xts, ccxts_ctx* _ctx, size_t key_nbytes, const void* data_key, const void* tweak_key) {
	CC_STUB_ERR();
};

void ccmode_xts_key_sched(const struct ccmode_xts* xts, ccxts_ctx* _ctx, size_t key_nbytes, const void* data_key, const void* tweak_key) {
	CC_STUB_VOID();
};

void* ccmode_xts_crypt(const ccxts_ctx* _ctx, ccxts_tweak* tweak, size_t nblocks, const void* in, void* out) {
	CC_STUB(NULL);
};

int ccmode_xts_set_tweak(const ccxts_ctx* _ctx, ccxts_tweak* tweak, const void* iv) {
	CC_STUB_ERR();
};
