/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

static void
dataerror_mark(void *ptr)
{
    rb_gc_mark((VALUE)ptr);
}

static void
dataerror_free(void *ptr)
{
}

static const rb_data_type_t dataerror_type = {
    "Bug #9167",
    {dataerror_mark, dataerror_free},
};

static VALUE
dataerror_alloc(VALUE klass)
{
    VALUE n = rb_str_new_cstr("[Bug #9167] error");
    return TypedData_Wrap_Struct(klass, &dataerror_type, (void *)n);
}

void
Init_dataerror(VALUE klass)
{
    VALUE rb_eDataErr = rb_define_class_under(klass, "DataError", rb_eStandardError);
    rb_define_alloc_func(rb_eDataErr, dataerror_alloc);
}
