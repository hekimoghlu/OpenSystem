/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "ruby/thread.h"

static void*
native_sleep_callback(void *data)
{
    struct timeval *timeval = data;
    select(0, NULL, NULL, NULL, timeval);

    return NULL;
}


static VALUE
thread_runnable_sleep(VALUE thread, VALUE timeout)
{
    struct timeval timeval;

    if (NIL_P(timeout)) {
	rb_raise(rb_eArgError, "timeout must be non nil");
    }

    timeval = rb_time_interval(timeout);

    rb_thread_call_without_gvl(native_sleep_callback, &timeval, RUBY_UBF_IO, NULL);

    return thread;
}

void
Init_call_without_gvl(void)
{
    rb_define_method(rb_cThread, "__runnable_sleep__", thread_runnable_sleep, 1);
}
