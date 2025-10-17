/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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
/* $Id: rmd160init.c 56456 2016-10-20 07:57:31Z svn $ */

#include <ruby/ruby.h>
#include "../digest.h"
#if defined(RMD160_USE_OPENSSL)
#include "rmd160ossl.h"
#else
#include "rmd160.h"
#endif

static const rb_digest_metadata_t rmd160 = {
    RUBY_DIGEST_API_VERSION,
    RMD160_DIGEST_LENGTH,
    RMD160_BLOCK_LENGTH,
    sizeof(RMD160_CTX),
    (rb_digest_hash_init_func_t)RMD160_Init,
    (rb_digest_hash_update_func_t)RMD160_Update,
    (rb_digest_hash_finish_func_t)RMD160_Finish,
};

/*
 * Document-class: Digest::RMD160 < Digest::Base
 * A class for calculating message digests using RIPEMD-160
 * cryptographic hash function, designed by Hans Dobbertin, Antoon
 * Bosselaers, and Bart Preneel.
 *
 * RMD160 calculates a digest of 160 bits (20 bytes).
 *
 * == Examples
 *  require 'digest'
 *
 *  # Compute a complete digest
 *  Digest::RMD160.hexdigest 'abc'      #=> "8eb208f7..."
 *
 *  # Compute digest by chunks
 *  rmd160 = Digest::RMD160.new               # =>#<Digest::RMD160>
 *  rmd160.update "ab"
 *  rmd160 << "c"                           # alias for #update
 *  rmd160.hexdigest                        # => "8eb208f7..."
 *
 *  # Use the same object to compute another digest
 *  rmd160.reset
 *  rmd160 << "message"
 *  rmd160.hexdigest                        # => "1dddbe1b..."
 */
void
Init_rmd160(void)
{
    VALUE mDigest, cDigest_Base, cDigest_RMD160;

    rb_require("digest");

#if 0
    mDigest = rb_define_module("Digest"); /* let rdoc know */
#endif
    mDigest = rb_path2class("Digest");
    cDigest_Base = rb_path2class("Digest::Base");

    cDigest_RMD160 = rb_define_class_under(mDigest, "RMD160", cDigest_Base);

#undef RUBY_UNTYPED_DATA_WARNING
#define RUBY_UNTYPED_DATA_WARNING 0
    rb_iv_set(cDigest_RMD160, "metadata",
	      Data_Wrap_Struct(0, 0, 0, (void *)&rmd160));
}
