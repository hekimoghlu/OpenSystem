/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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

static ID id_normal_ivar, id_internal_ivar;

static VALUE
init(VALUE self, VALUE arg1, VALUE arg2)
{
    rb_ivar_set(self, id_normal_ivar, arg1);
    rb_ivar_set(self, id_internal_ivar, arg2);
    return self;
}

static VALUE
get_normal(VALUE self)
{
    return rb_attr_get(self, id_normal_ivar);
}

static VALUE
get_internal(VALUE self)
{
    return rb_attr_get(self, id_internal_ivar);
}

void
Init_internal_ivar(void)
{
    VALUE mMarshal = rb_define_module_under(rb_define_module("Bug"), "Marshal");
    VALUE newclass = rb_define_class_under(mMarshal, "InternalIVar", rb_cObject);

    id_normal_ivar = rb_intern_const("normal");
#if 0
    /* leave id_internal_ivar being 0 */
    id_internal_ivar = rb_make_internal_id();
#endif
    rb_define_method(newclass, "initialize", init, 2);
    rb_define_method(newclass, "normal", get_normal, 0);
    rb_define_method(newclass, "internal", get_internal, 0);
}
