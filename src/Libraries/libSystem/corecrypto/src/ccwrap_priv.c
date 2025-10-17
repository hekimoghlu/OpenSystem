/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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

#include <corecrypto/ccwrap_priv.h>
#include <corecrypto/ccstubs.h>

int ccwrap_auth_decrypt_withiv(const struct ccmode_ecb* mode, ccecb_ctx* ctx, size_t wrappedLen, const void* wrappedKey, size_t* unwrappedLen, void* unwrappedKey, const void* iv) {
	CC_STUB_ERR();
};

int ccwrap_auth_encrypt_withiv(const struct ccmode_ecb* mode, ccecb_ctx* ctx, size_t unwrappedLen, const void* unwrappedKey, size_t* wrappedLen, void* wrappedKey, const void* iv) {
	CC_STUB_ERR();
};

size_t ccwrap_unwrapped_size(size_t wrappedLen) {
	CC_STUB_ERR();
};

size_t ccwrap_wrapped_size(size_t unwrappedLen) {
	CC_STUB_ERR();
};
