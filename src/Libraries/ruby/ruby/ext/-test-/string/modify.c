/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

VALUE
bug_str_modify(VALUE str)
{
    rb_str_modify(str);
    return str;
}

VALUE
bug_str_modify_expand(VALUE str, VALUE expand)
{
    rb_str_modify_expand(str, NUM2LONG(expand));
    return str;
}

void
Init_string_modify(VALUE klass)
{
    rb_define_method(klass, "modify!", bug_str_modify, 0);
    rb_define_method(klass, "modify_expand!", bug_str_modify_expand, 1);
}
