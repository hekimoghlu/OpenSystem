/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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

#include "internal.h"

static VALUE
str2big_poweroftwo(VALUE str, VALUE vbase, VALUE badcheck)
{
    return rb_str2big_poweroftwo(str, NUM2INT(vbase), RTEST(badcheck));
}

static VALUE
str2big_normal(VALUE str, VALUE vbase, VALUE badcheck)
{
    return rb_str2big_normal(str, NUM2INT(vbase), RTEST(badcheck));
}

static VALUE
str2big_karatsuba(VALUE str, VALUE vbase, VALUE badcheck)
{
    return rb_str2big_karatsuba(str, NUM2INT(vbase), RTEST(badcheck));
}

#if defined(HAVE_LIBGMP) && defined(HAVE_GMP_H)
static VALUE
str2big_gmp(VALUE str, VALUE vbase, VALUE badcheck)
{
    return rb_str2big_gmp(str, NUM2INT(vbase), RTEST(badcheck));
}
#else
#define str2big_gmp rb_f_notimplement
#endif

void
Init_str2big(VALUE klass)
{
    rb_define_method(rb_cString, "str2big_poweroftwo", str2big_poweroftwo, 2);
    rb_define_method(rb_cString, "str2big_normal", str2big_normal, 2);
    rb_define_method(rb_cString, "str2big_karatsuba", str2big_karatsuba, 2);
    rb_define_method(rb_cString, "str2big_gmp", str2big_gmp, 2);
}
