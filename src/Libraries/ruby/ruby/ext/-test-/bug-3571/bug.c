/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
bug_i(RB_BLOCK_CALL_FUNC_ARGLIST(i, arg))
{
    rb_notimplement();
    return ID2SYM(rb_frame_this_func());
}

static VALUE
bug_start(VALUE self, VALUE hash)
{
    VALUE ary = rb_ary_new3(1, Qnil);
    rb_block_call(ary, rb_intern("map"), 0, 0, bug_i, self);
    return ary;
}

void
Init_bug_3571(void)
{
    VALUE mBug = rb_define_module("Bug");
    rb_define_module_function(mBug, "start", bug_start, 0);
}
