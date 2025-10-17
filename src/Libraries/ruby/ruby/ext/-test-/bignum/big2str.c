/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
big(VALUE x)
{
    if (FIXNUM_P(x))
        return rb_int2big(FIX2LONG(x));
    if (RB_TYPE_P(x, T_BIGNUM))
        return x;
    rb_raise(rb_eTypeError, "can't convert %s to Bignum",
            rb_obj_classname(x));
}

static VALUE
big2str_generic(VALUE x, VALUE vbase)
{
    int base = NUM2INT(vbase);
    if (base < 2 || 36 < base)
        rb_raise(rb_eArgError, "invalid radix %d", base);
    return rb_big2str_generic(big(x), base);
}

#define POW2_P(x) (((x)&((x)-1))==0)

static VALUE
big2str_poweroftwo(VALUE x, VALUE vbase)
{
    int base = NUM2INT(vbase);
    if (base < 2 || 36 < base || !POW2_P(base))
        rb_raise(rb_eArgError, "invalid radix %d", base);
    return rb_big2str_poweroftwo(big(x), base);
}

#if defined(HAVE_LIBGMP) && defined(HAVE_GMP_H)
static VALUE
big2str_gmp(VALUE x, VALUE vbase)
{
    int base = NUM2INT(vbase);
    if (base < 2 || 36 < base)
        rb_raise(rb_eArgError, "invalid radix %d", base);
    return rb_big2str_gmp(big(x), base);
}
#else
#define big2str_gmp rb_f_notimplement
#endif

void
Init_big2str(VALUE klass)
{
    rb_define_method(rb_cInteger, "big2str_generic", big2str_generic, 1);
    rb_define_method(rb_cInteger, "big2str_poweroftwo", big2str_poweroftwo, 1);
    rb_define_method(rb_cInteger, "big2str_gmp", big2str_gmp, 1);
}
