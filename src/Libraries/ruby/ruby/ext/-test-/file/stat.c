/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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
#include "ruby/io.h"

static VALUE
stat_for_fd(VALUE self, VALUE fileno)
{
    struct stat st;
    if (fstat(NUM2INT(fileno), &st)) rb_sys_fail(0);
    return rb_stat_new(&st);
}

static VALUE
stat_for_path(VALUE self, VALUE path)
{
    struct stat st;
    FilePathValue(path);
    if (stat(RSTRING_PTR(path), &st)) rb_sys_fail(0);
    return rb_stat_new(&st);
}

void
Init_stat(VALUE module)
{
    VALUE st = rb_define_module_under(module, "Stat");
    rb_define_module_function(st, "for_fd", stat_for_fd, 1);
    rb_define_module_function(st, "for_path", stat_for_path, 1);
}
