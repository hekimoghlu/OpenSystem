/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
/* $Id: sha1init.c 56456 2016-10-20 07:57:31Z svn $ */

#include <ruby/ruby.h>
#include "../digest.h"
#if defined(SHA1_USE_OPENSSL)
#include "sha1ossl.h"
#elif defined(SHA1_USE_COMMONDIGEST)
#include "sha1cc.h"
#else
#include "sha1.h"
#endif

static const rb_digest_metadata_t sha1 = {
    RUBY_DIGEST_API_VERSION,
    SHA1_DIGEST_LENGTH,
    SHA1_BLOCK_LENGTH,
    sizeof(SHA1_CTX),
    (rb_digest_hash_init_func_t)SHA1_Init,
    (rb_digest_hash_update_func_t)SHA1_Update,
    (rb_digest_hash_finish_func_t)SHA1_Finish,
};

/*
 * Document-class: Digest::SHA1 < Digest::Base
 * A class for calculating message digests using the SHA-1 Secure Hash
 * Algorithm by NIST (the US' National Institute of Standards and
 * Technology), described in FIPS PUB 180-1.
 *
 * See Digest::Instance for digest API.
 *
 * SHA-1 calculates a digest of 160 bits (20 bytes).
 *
 * == Examples
 *  require 'digest'
 *
 *  # Compute a complete digest
 *  Digest::SHA1.hexdigest 'abc'      #=> "a9993e36..."
 *
 *  # Compute digest by chunks
 *  sha1 = Digest::SHA1.new               # =>#<Digest::SHA1>
 *  sha1.update "ab"
 *  sha1 << "c"                           # alias for #update
 *  sha1.hexdigest                        # => "a9993e36..."
 *
 *  # Use the same object to compute another digest
 *  sha1.reset
 *  sha1 << "message"
 *  sha1.hexdigest                        # => "6f9b9af3..."
 */
void
Init_sha1(void)
{
    VALUE mDigest, cDigest_Base, cDigest_SHA1;

    rb_require("digest");

#if 0
    mDigest = rb_define_module("Digest"); /* let rdoc know */
#endif
    mDigest = rb_path2class("Digest");
    cDigest_Base = rb_path2class("Digest::Base");

    cDigest_SHA1 = rb_define_class_under(mDigest, "SHA1", cDigest_Base);

#undef RUBY_UNTYPED_DATA_WARNING
#define RUBY_UNTYPED_DATA_WARNING 0
    rb_iv_set(cDigest_SHA1, "metadata",
	      Data_Wrap_Struct(0, 0, 0, (void *)&sha1));
}
