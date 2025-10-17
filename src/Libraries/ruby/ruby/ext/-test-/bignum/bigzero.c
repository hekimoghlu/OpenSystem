/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
bug_big_zero(VALUE self, VALUE length)
{
    long len = NUM2ULONG(length);
    VALUE z = rb_big_new(len, 1);
    MEMZERO(BIGNUM_DIGITS(z), BDIGIT, len);
    return z;
}

static VALUE
bug_big_negzero(VALUE self, VALUE length)
{
    long len = NUM2ULONG(length);
    VALUE z = rb_big_new(len, 0);
    MEMZERO(BIGNUM_DIGITS(z), BDIGIT, len);
    return z;
}

void
Init_bigzero(VALUE klass)
{
    rb_define_singleton_method(klass, "zero", bug_big_zero, 1);
    rb_define_singleton_method(klass, "negzero", bug_big_negzero, 1);
}
