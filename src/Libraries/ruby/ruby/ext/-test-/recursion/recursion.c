/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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

#include <ruby.h>

static VALUE
recursive_i(VALUE obj, VALUE mid, int recur)
{
    if (recur) return Qnil;
    return rb_funcallv(obj, rb_to_id(mid), 0, 0);
}

static VALUE
exec_recursive(VALUE self, VALUE mid)
{
    return rb_exec_recursive(recursive_i, self, mid);
}

static VALUE
exec_recursive_outer(VALUE self, VALUE mid)
{
    return rb_exec_recursive_outer(recursive_i, self, mid);
}

void
Init_recursion(void)
{
    VALUE m = rb_define_module_under(rb_define_module("Bug"), "Recursive");
    rb_define_method(m, "exec_recursive", exec_recursive, 1);
    rb_define_method(m, "exec_recursive_outer", exec_recursive_outer, 1);
}
