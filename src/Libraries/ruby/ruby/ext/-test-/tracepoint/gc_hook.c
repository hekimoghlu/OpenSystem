/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

static int invoking; /* TODO: should not be global variable */

static VALUE
invoke_proc_ensure(void *dmy)
{
    invoking = 0;
    return Qnil;
}

static VALUE
invoke_proc_begin(VALUE proc)
{
    return rb_proc_call(proc, rb_ary_new());
}

static void
invoke_proc(void *data)
{
    VALUE proc = (VALUE)data;
    invoking += 1;
    rb_ensure(invoke_proc_begin, proc, invoke_proc_ensure, 0);
}

static void
gc_start_end_i(VALUE tpval, void *data)
{
    if (0) {
	rb_trace_arg_t *tparg = rb_tracearg_from_tracepoint(tpval);
	fprintf(stderr, "trace: %s\n", rb_tracearg_event_flag(tparg) == RUBY_INTERNAL_EVENT_GC_START ? "gc_start" : "gc_end");
    }

    if (invoking == 0) {
	rb_postponed_job_register(0, invoke_proc, data);
    }
}

static VALUE
set_gc_hook(VALUE module, VALUE proc, rb_event_flag_t event, const char *tp_str, const char *proc_str)
{
    VALUE tpval;
    ID tp_key = rb_intern(tp_str);
    ID proc_key = rb_intern(proc_str);

    /* disable previous keys */
    if (rb_ivar_defined(module, tp_key) != 0 &&
	RTEST(tpval = rb_ivar_get(module, tp_key))) {
	rb_tracepoint_disable(tpval);
	rb_ivar_set(module, tp_key, Qnil);
	rb_ivar_set(module, proc_key, Qnil);
    }

    if (RTEST(proc)) {
	if (!rb_obj_is_proc(proc)) {
	    rb_raise(rb_eTypeError, "trace_func needs to be Proc");
	}

	tpval = rb_tracepoint_new(0, event, gc_start_end_i, (void *)proc);
	rb_ivar_set(module, tp_key, tpval);
	rb_ivar_set(module, proc_key, proc); /* GC guard */
	rb_tracepoint_enable(tpval);
    }

    return proc;
}

static VALUE
set_after_gc_start(VALUE module, VALUE proc)
{
    return set_gc_hook(module, proc, RUBY_INTERNAL_EVENT_GC_START,
		       "__set_after_gc_start_tpval__", "__set_after_gc_start_proc__");
}

void
Init_gc_hook(VALUE module)
{
    rb_define_module_function(module, "after_gc_start_hook=", set_after_gc_start, 1);
}
