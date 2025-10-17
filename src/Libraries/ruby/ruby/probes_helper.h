/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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

#ifndef RUBY_PROBES_HELPER_H
#define RUBY_PROBES_HELPER_H

#include "ruby/ruby.h"

struct ruby_dtrace_method_hook_args {
    const char *classname;
    const char *methodname;
    const char *filename;
    int line_no;
    volatile VALUE klass;
    volatile VALUE name;
};

NOINLINE(int rb_dtrace_setup(rb_execution_context_t *, VALUE, ID, struct ruby_dtrace_method_hook_args *));

#define RUBY_DTRACE_METHOD_HOOK(name, ec, klazz, id) \
do { \
    if (UNLIKELY(RUBY_DTRACE_##name##_ENABLED())) { \
	struct ruby_dtrace_method_hook_args args; \
	if (rb_dtrace_setup(ec, klazz, id, &args)) { \
	    RUBY_DTRACE_##name(args.classname, \
			       args.methodname, \
			       args.filename, \
			       args.line_no); \
	} \
    } \
} while (0)

#define RUBY_DTRACE_METHOD_ENTRY_HOOK(ec, klass, id) \
    RUBY_DTRACE_METHOD_HOOK(METHOD_ENTRY, ec, klass, id)

#define RUBY_DTRACE_METHOD_RETURN_HOOK(ec, klass, id) \
    RUBY_DTRACE_METHOD_HOOK(METHOD_RETURN, ec, klass, id)

#define RUBY_DTRACE_CMETHOD_ENTRY_HOOK(ec, klass, id) \
    RUBY_DTRACE_METHOD_HOOK(CMETHOD_ENTRY, ec, klass, id)

#define RUBY_DTRACE_CMETHOD_RETURN_HOOK(ec, klass, id) \
    RUBY_DTRACE_METHOD_HOOK(CMETHOD_RETURN, ec, klass, id)

#endif /* RUBY_PROBES_HELPER_H */
