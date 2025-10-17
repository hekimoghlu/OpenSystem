/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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

#include "ruby.h"
#include "ruby/encoding.h"

VALUE
bug_str_enc_associate(VALUE str, VALUE enc)
{
    return rb_enc_associate(str, rb_to_encoding(enc));
}

VALUE
bug_str_encoding_index(VALUE self, VALUE str)
{
    int idx = rb_enc_get_index(str);
    return INT2NUM(idx);
}

void
Init_string_enc_associate(VALUE klass)
{
    rb_define_method(klass, "associate_encoding!", bug_str_enc_associate, 1);
    rb_define_singleton_method(klass, "encoding_index", bug_str_encoding_index, 1);
}
