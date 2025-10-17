/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
usr_dumper(VALUE self)
{
    return self;
}

static VALUE
usr_loader(VALUE self, VALUE m)
{
    VALUE val = rb_ivar_get(m, rb_intern("@value"));
    *(int *)DATA_PTR(self) = NUM2INT(val);
    return self;
}

static VALUE
compat_mload(VALUE self, VALUE data)
{
    rb_ivar_set(self, rb_intern("@value"), data);
    return self;
}

void
Init_compat(void)
{
    VALUE newclass = rb_path2class("Bug::Marshal::UsrMarshal");
    VALUE oldclass = rb_define_class_under(newclass, "compat", rb_cObject);

    rb_define_method(oldclass, "marshal_load", compat_mload, 1);
    rb_marshal_define_compat(newclass, oldclass, usr_dumper, usr_loader);
}
