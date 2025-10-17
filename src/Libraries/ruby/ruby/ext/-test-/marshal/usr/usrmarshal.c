/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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

static size_t
usr_size(const void *ptr)
{
    return sizeof(int);
}

static const rb_data_type_t usrmarshal_type = {
    "UsrMarshal",
    {0, RUBY_DEFAULT_FREE, usr_size,},
    0, 0,
    RUBY_TYPED_FREE_IMMEDIATELY|RUBY_TYPED_WB_PROTECTED,
};

static VALUE
usr_alloc(VALUE klass)
{
    int *p;
    return TypedData_Make_Struct(klass, int, &usrmarshal_type, p);
}

static VALUE
usr_init(VALUE self, VALUE val)
{
    int *ptr = Check_TypedStruct(self, &usrmarshal_type);
    *ptr = NUM2INT(val);
    return self;
}

static VALUE
usr_value(VALUE self)
{
    int *ptr = Check_TypedStruct(self, &usrmarshal_type);
    int val = *ptr;
    return INT2NUM(val);
}

void
Init_usr(void)
{
    VALUE mMarshal = rb_define_module_under(rb_define_module("Bug"), "Marshal");
    VALUE newclass = rb_define_class_under(mMarshal, "UsrMarshal", rb_cObject);

    rb_define_alloc_func(newclass, usr_alloc);
    rb_define_method(newclass, "initialize", usr_init, 1);
    rb_define_method(newclass, "value", usr_value, 0);
    rb_define_method(newclass, "marshal_load", usr_init, 1);
    rb_define_method(newclass, "marshal_dump", usr_value, 0);
}
