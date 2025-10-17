/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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

#ifndef SHA2OSSL_H_INCLUDED
#define SHA2OSSL_H_INCLUDED

#include <stddef.h>
#include <openssl/sha.h>

#define SHA256_BLOCK_LENGTH	SHA256_CBLOCK
#define SHA384_BLOCK_LENGTH	SHA512_CBLOCK
#define SHA512_BLOCK_LENGTH	SHA512_CBLOCK

#ifndef __DragonFly__
#define SHA384_Final SHA512_Final
#endif

typedef SHA512_CTX SHA384_CTX;

#undef SHA256_Finish
#undef SHA384_Finish
#undef SHA512_Finish
#define SHA256_Finish rb_digest_SHA256_finish
#define SHA384_Finish rb_digest_SHA384_finish
#define SHA512_Finish rb_digest_SHA512_finish
static DEFINE_FINISH_FUNC_FROM_FINAL(SHA256)
static DEFINE_FINISH_FUNC_FROM_FINAL(SHA384)
static DEFINE_FINISH_FUNC_FROM_FINAL(SHA512)

#endif
