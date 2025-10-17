/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
bug_funcall(int argc, VALUE *argv, VALUE self)
{
    if (argc < 1) rb_raise(rb_eArgError, "not enough argument");
    return rb_funcallv(self, rb_to_id(*argv), argc-1, argv+1);
}

void
Init_notimplement(void)
{
    VALUE mBug = rb_define_module("Bug");
    VALUE klass = rb_define_class_under(mBug, "NotImplement", rb_cObject);
    rb_define_module_function(mBug, "funcall", bug_funcall, -1);
    rb_define_module_function(mBug, "notimplement", rb_f_notimplement, -1);
    rb_define_method(klass, "notimplement", rb_f_notimplement, -1);
}
