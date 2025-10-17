/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
/* $Id: sha2init.c 52694 2015-11-21 04:35:57Z naruse $ */

#include <ruby/ruby.h>
#include "../digest.h"
#if defined(SHA2_USE_OPENSSL)
#include "sha2ossl.h"
#elif defined(SHA2_USE_COMMONDIGEST)
#include "sha2cc.h"
#else
#include "sha2.h"
#endif

#define FOREACH_BITLEN(func)	func(256) func(384) func(512)

#define DEFINE_ALGO_METADATA(bitlen) \
static const rb_digest_metadata_t sha##bitlen = { \
    RUBY_DIGEST_API_VERSION, \
    SHA##bitlen##_DIGEST_LENGTH, \
    SHA##bitlen##_BLOCK_LENGTH, \
    sizeof(SHA##bitlen##_CTX), \
    (rb_digest_hash_init_func_t)SHA##bitlen##_Init, \
    (rb_digest_hash_update_func_t)SHA##bitlen##_Update, \
    (rb_digest_hash_finish_func_t)SHA##bitlen##_Finish, \
};

FOREACH_BITLEN(DEFINE_ALGO_METADATA)

/*
 * Classes for calculating message digests using the SHA-256/384/512
 * Secure Hash Algorithm(s) by NIST (the US' National Institute of
 * Standards and Technology), described in FIPS PUB 180-2.
 */
void
Init_sha2(void)
{
    VALUE mDigest, cDigest_Base;
    ID id_metadata;

#define DECLARE_ALGO_CLASS(bitlen) \
    VALUE cDigest_SHA##bitlen;

    FOREACH_BITLEN(DECLARE_ALGO_CLASS)

    rb_require("digest");

    id_metadata = rb_intern_const("metadata");

    mDigest = rb_path2class("Digest");
    cDigest_Base = rb_path2class("Digest::Base");

#define DEFINE_ALGO_CLASS(bitlen) \
    cDigest_SHA##bitlen = rb_define_class_under(mDigest, "SHA" #bitlen, cDigest_Base); \
\
    rb_ivar_set(cDigest_SHA##bitlen, id_metadata, \
		Data_Wrap_Struct(0, 0, 0, (void *)&sha##bitlen));

#undef RUBY_UNTYPED_DATA_WARNING
#define RUBY_UNTYPED_DATA_WARNING 0
    FOREACH_BITLEN(DEFINE_ALGO_CLASS)
}
