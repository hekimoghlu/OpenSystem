/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#include "ruby/ruby.h"
#include "ruby/debug.h"

static VALUE
callback(const rb_debug_inspector_t *dbg_context, void *data)
{
    VALUE locs = rb_debug_inspector_backtrace_locations(dbg_context);
    long i, len = RARRAY_LEN(locs);
    VALUE binds = rb_ary_new();
    for (i = 0; i < len; ++i) {
	VALUE entry = rb_ary_new();
	rb_ary_push(binds, entry);
	rb_ary_push(entry, rb_debug_inspector_frame_self_get(dbg_context, i));
	rb_ary_push(entry, rb_debug_inspector_frame_binding_get(dbg_context, i));
	rb_ary_push(entry, rb_debug_inspector_frame_class_get(dbg_context, i));
	rb_ary_push(entry, rb_debug_inspector_frame_iseq_get(dbg_context, i));
	rb_ary_push(entry, rb_ary_entry(locs, i));
    }
    return binds;
}

static VALUE
debug_inspector(VALUE self)
{
    return rb_debug_inspector_open(callback, NULL);
}

void
Init_inspector(VALUE klass)
{
    rb_define_module_function(klass, "inspector", debug_inspector, 0);
}
