/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
load_protect(int argc, VALUE *argv, VALUE self)
{
    int state;
    VALUE path, wrap;
    rb_scan_args(argc, argv, "11", &path, &wrap);
    rb_load_protect(path, RTEST(wrap), &state);
    if (state) rb_jump_tag(state);
    return Qnil;
}

void
Init_protect(void)
{
    VALUE mod = rb_define_module("Bug");
    rb_define_singleton_method(mod, "load_protect", load_protect, -1);
}
