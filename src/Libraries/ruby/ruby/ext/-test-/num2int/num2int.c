/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
test_num2short(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%d", NUM2SHORT(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_num2ushort(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%u", NUM2USHORT(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_num2int(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%d", NUM2INT(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_num2uint(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%u", NUM2UINT(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_num2long(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%ld", NUM2LONG(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_num2ulong(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%lu", NUM2ULONG(num));
    return rb_str_new_cstr(buf);
}

#ifdef HAVE_LONG_LONG
static VALUE
test_num2ll(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%"PRI_LL_PREFIX"d", NUM2LL(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_num2ull(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%"PRI_LL_PREFIX"u", NUM2ULL(num));
    return rb_str_new_cstr(buf);
}
#endif

static VALUE
test_fix2short(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%d", FIX2SHORT(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_fix2int(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%d", FIX2INT(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_fix2uint(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%u", FIX2UINT(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_fix2long(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%ld", FIX2LONG(num));
    return rb_str_new_cstr(buf);
}

static VALUE
test_fix2ulong(VALUE obj, VALUE num)
{
    char buf[128];
    sprintf(buf, "%lu", FIX2ULONG(num));
    return rb_str_new_cstr(buf);
}

void
Init_num2int(void)
{
    VALUE mNum2int = rb_define_module("Num2int");

    rb_define_module_function(mNum2int, "NUM2SHORT", test_num2short, 1);
    rb_define_module_function(mNum2int, "NUM2USHORT", test_num2ushort, 1);

    rb_define_module_function(mNum2int, "NUM2INT", test_num2int, 1);
    rb_define_module_function(mNum2int, "NUM2UINT", test_num2uint, 1);

    rb_define_module_function(mNum2int, "NUM2LONG", test_num2long, 1);
    rb_define_module_function(mNum2int, "NUM2ULONG", test_num2ulong, 1);

#ifdef HAVE_LONG_LONG
    rb_define_module_function(mNum2int, "NUM2LL", test_num2ll, 1);
    rb_define_module_function(mNum2int, "NUM2ULL", test_num2ull, 1);
#endif

    rb_define_module_function(mNum2int, "FIX2SHORT", test_fix2short, 1);

    rb_define_module_function(mNum2int, "FIX2INT", test_fix2int, 1);
    rb_define_module_function(mNum2int, "FIX2UINT", test_fix2uint, 1);

    rb_define_module_function(mNum2int, "FIX2LONG", test_fix2long, 1);
    rb_define_module_function(mNum2int, "FIX2ULONG", test_fix2ulong, 1);
}

