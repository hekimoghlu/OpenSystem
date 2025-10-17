/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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
mul_normal(VALUE x, VALUE y)
{
    return rb_big_norm(rb_big_mul_normal(big(x), big(y)));
}

static VALUE
sq_fast(VALUE x)
{
    return rb_big_norm(rb_big_sq_fast(big(x)));
}

static VALUE
mul_balance(VALUE x, VALUE y)
{
    return rb_big_norm(rb_big_mul_balance(big(x), big(y)));
}

static VALUE
mul_karatsuba(VALUE x, VALUE y)
{
    return rb_big_norm(rb_big_mul_karatsuba(big(x), big(y)));
}

static VALUE
mul_toom3(VALUE x, VALUE y)
{
    return rb_big_norm(rb_big_mul_toom3(big(x), big(y)));
}

#if defined(HAVE_LIBGMP) && defined(HAVE_GMP_H)
static VALUE
mul_gmp(VALUE x, VALUE y)
{
    return rb_big_norm(rb_big_mul_gmp(big(x), big(y)));
}
#else
#define mul_gmp rb_f_notimplement
#endif

void
Init_mul(VALUE klass)
{
    rb_define_const(rb_cInteger, "SIZEOF_BDIGIT", INT2NUM(SIZEOF_BDIGIT));
    rb_define_const(rb_cInteger, "BITSPERDIG", INT2NUM(SIZEOF_BDIGIT * CHAR_BIT));
    rb_define_method(rb_cInteger, "big_mul_normal", mul_normal, 1);
    rb_define_method(rb_cInteger, "big_sq_fast", sq_fast, 0);
    rb_define_method(rb_cInteger, "big_mul_balance", mul_balance, 1);
    rb_define_method(rb_cInteger, "big_mul_karatsuba", mul_karatsuba, 1);
    rb_define_method(rb_cInteger, "big_mul_toom3", mul_toom3, 1);
    rb_define_method(rb_cInteger, "big_mul_gmp", mul_gmp, 1);
}
