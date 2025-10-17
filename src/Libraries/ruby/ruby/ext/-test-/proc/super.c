/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
bug_proc_call_super(RB_BLOCK_CALL_FUNC_ARGLIST(yieldarg, procarg))
{
    VALUE args[2];
    VALUE ret;
    args[0] = yieldarg;
    args[1] = procarg;
    ret = rb_call_super(2, args);
    if (!NIL_P(blockarg)) {
	ret = rb_proc_call(blockarg, ret);
    }
    return ret;
}

static VALUE
bug_proc_make_call_super(VALUE self, VALUE procarg)
{
    return rb_proc_new(bug_proc_call_super, procarg);
}

void
Init_super(VALUE klass)
{
    rb_define_singleton_method(klass, "make_call_super", bug_proc_make_call_super, 1);
}
