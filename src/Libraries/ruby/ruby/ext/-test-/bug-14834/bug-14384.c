/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

#include <ruby/ruby.h>
#include <ruby/debug.h>

#ifndef MAYBE_UNUSED
# define MAYBE_UNUSED(x) x
#endif

static NOINLINE(VALUE f(VALUE));
static NOINLINE(void g(VALUE, void*));
extern NOINLINE(void Init_bug_14384(void));

void
Init_bug_14834(void)
{
    VALUE q = rb_define_module("Bug");
    rb_define_module_function(q, "bug_14834", f, 0);
}

VALUE
f(VALUE q)
{
    int   w[] = { 0, 1024 };
    VALUE e   = rb_tracepoint_new(Qnil, RUBY_INTERNAL_EVENT_NEWOBJ, g, w);

    rb_tracepoint_enable(e);
    return rb_ensure(rb_yield, q, rb_tracepoint_disable, e);
}

void
g(MAYBE_UNUSED(VALUE q), void* w)
{
    const int *e = (const int *)w;
    const int  r = *e++;
    const int  t = *e++;
    VALUE     *y = ALLOCA_N(VALUE, t);
    int       *u = ALLOCA_N(int, t);

    rb_profile_frames(r, t, y, u);
}
