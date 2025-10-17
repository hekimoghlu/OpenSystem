/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#ifndef SHA1OSSL_H_INCLUDED
#define SHA1OSSL_H_INCLUDED

#include <stddef.h>
#include <openssl/sha.h>

#define SHA1_CTX	SHA_CTX

#ifdef SHA_BLOCK_LENGTH
#define SHA1_BLOCK_LENGTH	SHA_BLOCK_LENGTH
#else
#define SHA1_BLOCK_LENGTH	SHA_CBLOCK
#endif
#define SHA1_DIGEST_LENGTH	SHA_DIGEST_LENGTH

static DEFINE_FINISH_FUNC_FROM_FINAL(SHA1)
#undef SHA1_Finish
#define SHA1_Finish rb_digest_SHA1_finish

#endif
