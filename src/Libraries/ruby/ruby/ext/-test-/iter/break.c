/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
iter_break(VALUE self)
{
    rb_iter_break();

    UNREACHABLE_RETURN(Qnil);
}

static VALUE
iter_break_value(VALUE self, VALUE val)
{
    rb_iter_break_value(val);

    UNREACHABLE_RETURN(Qnil);
}

void
Init_break(VALUE klass)
{
    VALUE breakable = rb_define_module_under(klass, "Breakable");
    rb_define_module_function(breakable, "iter_break", iter_break, 0);
    rb_define_module_function(breakable, "iter_break_value", iter_break_value, 1);
}
