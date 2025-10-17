/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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
obj_method_arity(VALUE self, VALUE obj, VALUE mid)
{
    int arity = rb_obj_method_arity(obj, rb_check_id(&mid));
    return INT2FIX(arity);
}

static VALUE
mod_method_arity(VALUE self, VALUE mod, VALUE mid)
{
    int arity = rb_mod_method_arity(mod, rb_check_id(&mid));
    return INT2FIX(arity);
}

void
Init_arity(VALUE mod)
{
    rb_define_module_function(mod, "obj_method_arity", obj_method_arity, 2);
    rb_define_module_function(mod, "mod_method_arity", mod_method_arity, 2);
}
