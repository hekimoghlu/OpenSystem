/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#ifndef RMD160OSSL_H_INCLUDED
#define RMD160OSSL_H_INCLUDED

#include <stddef.h>
#include <openssl/ripemd.h>

#define RMD160_CTX	RIPEMD160_CTX

#define RMD160_Init	RIPEMD160_Init
#define RMD160_Update	RIPEMD160_Update

#define RMD160_BLOCK_LENGTH		RIPEMD160_CBLOCK
#define RMD160_DIGEST_LENGTH		RIPEMD160_DIGEST_LENGTH

static DEFINE_FINISH_FUNC_FROM_FINAL(RIPEMD160)
#define RMD160_Finish rb_digest_RIPEMD160_finish

#endif
