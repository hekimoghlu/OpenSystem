/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
divrem_normal(VALUE x, VALUE y)
{
    return rb_big_norm(rb_big_divrem_normal(big(x), big(y)));
}

#if defined(HAVE_LIBGMP) && defined(HAVE_GMP_H)
static VALUE
divrem_gmp(VALUE x, VALUE y)
{
    return rb_big_norm(rb_big_divrem_gmp(big(x), big(y)));
}
#else
#define divrem_gmp rb_f_notimplement
#endif

void
Init_div(VALUE klass)
{
    rb_define_method(rb_cInteger, "big_divrem_normal", divrem_normal, 1);
    rb_define_method(rb_cInteger, "big_divrem_gmp", divrem_gmp, 1);
}
