/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
int_bignum_p(VALUE self)
{
    return RB_TYPE_P(self, T_BIGNUM) ? Qtrue : Qfalse;
}

static VALUE
int_fixnum_p(VALUE self)
{
    return FIXNUM_P(self) ? Qtrue : Qfalse;
}

static VALUE
rb_int_to_bignum(VALUE x)
{
    if (FIXNUM_P(x))
        x = rb_int2big(FIX2LONG(x));
    return x;
}

static VALUE
positive_pow(VALUE x, VALUE y)
{
    return rb_int_positive_pow(NUM2LONG(x), NUM2ULONG(y));
}

void
Init_core_ext(VALUE klass)
{
    rb_define_method(rb_cInteger, "bignum?", int_bignum_p, 0);
    rb_define_method(rb_cInteger, "fixnum?", int_fixnum_p, 0);
    rb_define_method(rb_cInteger, "to_bignum", rb_int_to_bignum, 0);
    rb_define_method(rb_cInteger, "positive_pow", positive_pow, 1);
}
