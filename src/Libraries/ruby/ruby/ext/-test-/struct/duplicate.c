/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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

static VALUE
bug_struct_new_duplicate(VALUE obj, VALUE name, VALUE mem)
{
    const char *n = NIL_P(name) ? 0 : StringValueCStr(name);
    const char *m = StringValueCStr(mem);
    return rb_struct_define(n, m, m, NULL);
}

static VALUE
bug_struct_new_duplicate_under(VALUE obj, VALUE name, VALUE mem)
{
    const char *n = StringValueCStr(name);
    const char *m = StringValueCStr(mem);
    return rb_struct_define_under(obj, n, m, m, NULL);
}

void
Init_duplicate(VALUE klass)
{
    rb_define_singleton_method(klass, "new_duplicate", bug_struct_new_duplicate, 2);
    rb_define_singleton_method(klass, "new_duplicate_under", bug_struct_new_duplicate_under, 2);
}
