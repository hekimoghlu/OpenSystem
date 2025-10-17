/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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
#include <dlfcn.h>

struct data_for_loop_dlsym {
    const char *name;
    volatile int stop;
};

static void*
native_loop_dlsym(void *data)
{
    struct data_for_loop_dlsym *s = data;

    while (!(s->stop)) {
        dlsym(RTLD_DEFAULT, s->name);
    }

    return NULL;
}

static void
ubf_for_loop_dlsym(void *data)
{
    struct data_for_loop_dlsym *s = data;

    s->stop = 1;

    return;
}

static VALUE
loop_dlsym(VALUE self, VALUE name)
{
    struct data_for_loop_dlsym d;

    d.stop = 0;
    d.name = StringValuePtr(name);

    rb_thread_call_without_gvl(native_loop_dlsym, &d,
                               ubf_for_loop_dlsym, &d);

    return self;
}

void
Init_infinite_loop_dlsym(void)
{
    rb_define_method(rb_cThread, "__infinite_loop_dlsym__", loop_dlsym, 1);
}
