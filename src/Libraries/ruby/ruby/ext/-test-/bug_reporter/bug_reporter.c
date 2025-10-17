/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include <stdio.h>

int rb_bug_reporter_add(void (*func)(FILE *, void *), void *data);

static void
sample_bug_reporter(FILE *out, void *ptr)
{
    int n = (int)(uintptr_t)ptr;
    fprintf(out, "Sample bug reporter: %d\n", n);
}

static VALUE
register_sample_bug_reporter(VALUE self, VALUE obj)
{
    rb_bug_reporter_add(sample_bug_reporter, (void *)(uintptr_t)NUM2INT(obj));
    return Qnil;
}

void
Init_bug_reporter(void)
{
    rb_define_global_function("register_sample_bug_reporter", register_sample_bug_reporter, 1);
}
