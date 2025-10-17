/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

VALUE rb_funcall_passing_block(VALUE, ID, int, const VALUE*);

static VALUE
with_funcall2(int argc, VALUE *argv, VALUE self)
{
    return rb_funcallv(self, rb_intern("target"), argc, argv);
}

static VALUE
with_funcall_passing_block(int argc, VALUE *argv, VALUE self)
{
    return rb_funcall_passing_block(self, rb_intern("target"), argc, argv);
}

static VALUE
extra_args_name(VALUE self)
{
    /*
     * at least clang 5.x gets tripped by the extra 0 arg
     * [ruby-core:85266] [Bug #14425]
     */
    return rb_funcall(self, rb_intern("name"), 0, 0);
}

void
Init_funcall(void)
{
    VALUE cTestFuncall = rb_path2class("TestFuncall");
    VALUE cRelay = rb_define_module_under(cTestFuncall, "Relay");

    rb_define_singleton_method(cRelay,
			       "with_funcall2",
			       with_funcall2,
			       -1);
    rb_define_singleton_method(cRelay,
			       "with_funcall_passing_block",
			       with_funcall_passing_block,
			       -1);
    rb_define_singleton_method(cTestFuncall, "extra_args_name",
                                extra_args_name,
                                0);
}
