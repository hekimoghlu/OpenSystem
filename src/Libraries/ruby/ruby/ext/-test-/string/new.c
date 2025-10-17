/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

static VALUE
bug_str_buf_new(VALUE self, VALUE len)
{
    return rb_str_buf_new(NUM2LONG(len));
}

static VALUE
bug_external_str_new(VALUE self, VALUE len, VALUE enc)
{
    return rb_external_str_new_with_enc(NULL, NUM2LONG(len), rb_to_encoding(enc));
}

void
Init_string_new(VALUE klass)
{
    rb_define_singleton_method(klass, "buf_new", bug_str_buf_new, 1);
    rb_define_singleton_method(klass, "external_new", bug_external_str_new, 2);
}
