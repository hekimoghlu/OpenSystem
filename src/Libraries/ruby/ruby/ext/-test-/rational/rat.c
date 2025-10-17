/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#if defined(HAVE_LIBGMP) && defined(HAVE_GMP_H)
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
#endif

static VALUE
gcd_normal(VALUE x, VALUE y)
{
    return rb_big_norm(rb_gcd_normal(rb_to_int(x), rb_to_int(y)));
}

#if defined(HAVE_LIBGMP) && defined(HAVE_GMP_H)
static VALUE
gcd_gmp(VALUE x, VALUE y)
{
    return rb_big_norm(rb_gcd_gmp(big(x), big(y)));
}
#else
#define gcd_gmp rb_f_notimplement
#endif

void
Init_rational(VALUE klass)
{
    rb_define_method(rb_cInteger, "gcd_normal", gcd_normal, 1);
    rb_define_method(rb_cInteger, "gcd_gmp", gcd_gmp, 1);
}
