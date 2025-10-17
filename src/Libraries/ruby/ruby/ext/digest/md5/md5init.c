/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
/* $Id: md5init.c 56459 2016-10-20 13:18:52Z kazu $ */

#include <ruby/ruby.h>
#include "../digest.h"
#if defined(MD5_USE_OPENSSL)
#include "md5ossl.h"
#elif defined(MD5_USE_COMMONDIGEST)
#include "md5cc.h"
#else
#include "md5.h"
#endif

static const rb_digest_metadata_t md5 = {
    RUBY_DIGEST_API_VERSION,
    MD5_DIGEST_LENGTH,
    MD5_BLOCK_LENGTH,
    sizeof(MD5_CTX),
    (rb_digest_hash_init_func_t)MD5_Init,
    (rb_digest_hash_update_func_t)MD5_Update,
    (rb_digest_hash_finish_func_t)MD5_Finish,
};

/*
 * Document-class: Digest::MD5 < Digest::Base
 * A class for calculating message digests using the MD5
 * Message-Digest Algorithm by RSA Data Security, Inc., described in
 * RFC1321.
 *
 * MD5 calculates a digest of 128 bits (16 bytes).
 *
 * == Examples
 *  require 'digest'
 *
 *  # Compute a complete digest
 *  Digest::MD5.hexdigest 'abc'      #=> "90015098..."
 *
 *  # Compute digest by chunks
 *  md5 = Digest::MD5.new               # =>#<Digest::MD5>
 *  md5.update "ab"
 *  md5 << "c"                           # alias for #update
 *  md5.hexdigest                        # => "90015098..."
 *
 *  # Use the same object to compute another digest
 *  md5.reset
 *  md5 << "message"
 *  md5.hexdigest                        # => "78e73102..."
 */
void
Init_md5(void)
{
    VALUE mDigest, cDigest_Base, cDigest_MD5;

    rb_require("digest");

#if 0
    mDigest = rb_define_module("Digest"); /* let rdoc know */
#endif
    mDigest = rb_path2class("Digest");
    cDigest_Base = rb_path2class("Digest::Base");

    cDigest_MD5 = rb_define_class_under(mDigest, "MD5", cDigest_Base);

#undef RUBY_UNTYPED_DATA_WARNING
#define RUBY_UNTYPED_DATA_WARNING 0
    rb_iv_set(cDigest_MD5, "metadata",
	      Data_Wrap_Struct(0, 0, 0, (void *)&md5));
}
