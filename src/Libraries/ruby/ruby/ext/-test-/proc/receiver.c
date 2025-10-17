/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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

VALUE rb_current_receiver(void);

static VALUE
bug_proc_call_receiver(RB_BLOCK_CALL_FUNC_ARGLIST(yieldarg, procarg))
{
    return rb_current_receiver();
}

static VALUE
bug_proc_make_call_receiver(VALUE self, VALUE procarg)
{
    return rb_proc_new(bug_proc_call_receiver, procarg);
}

void
Init_receiver(VALUE klass)
{
    rb_define_singleton_method(klass, "make_call_receiver", bug_proc_make_call_receiver, 1);
}
