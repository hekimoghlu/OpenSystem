/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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

VALUE rb_str_dup(VALUE str);

static VALUE
bug_rb_str_dup(VALUE self, VALUE str)
{
    rb_check_type(str, T_STRING);
    return rb_str_dup(str);
}

static VALUE
bug_shared_string_p(VALUE self, VALUE str)
{
    rb_check_type(str, T_STRING);
    return RB_FL_TEST(str, RUBY_ELTS_SHARED) && RB_FL_TEST(str, RSTRING_NOEMBED) ? Qtrue : Qfalse;
}

static VALUE
bug_sharing_with_shared_p(VALUE self, VALUE str)
{
    rb_check_type(str, T_STRING);
    if (bug_shared_string_p(self, str)) {
        return bug_shared_string_p(self, RSTRING(str)->as.heap.aux.shared);
    }
    return Qfalse;
}

void
Init_string_rb_str_dup(VALUE klass)
{
    rb_define_singleton_method(klass, "rb_str_dup", bug_rb_str_dup, 1);
    rb_define_singleton_method(klass, "shared_string?", bug_shared_string_p, 1);
    rb_define_singleton_method(klass, "sharing_with_shared?", bug_sharing_with_shared_p, 1);
}
