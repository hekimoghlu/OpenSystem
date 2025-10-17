/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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

#include "ruby.h"
#include "ruby/util.h"
#include "ruby/encoding.h"

struct sort_data {
    rb_encoding *enc;
    long elsize;
};

static int
cmp_1(const void *ap, const void *bp, void *dummy)
{
    struct sort_data *d = dummy;
    VALUE a = rb_enc_str_new(ap, d->elsize, d->enc);
    VALUE b = rb_enc_str_new(bp, d->elsize, d->enc);
    VALUE retval = rb_yield_values(2, a, b);
    return rb_cmpint(retval, a, b);
}

static int
cmp_2(const void *ap, const void *bp, void *dummy)
{
    int a = *(const unsigned char *)ap;
    int b = *(const unsigned char *)bp;
    return a - b;
}

static VALUE
bug_str_qsort_bang(int argc, VALUE *argv, VALUE str)
{
    VALUE beg, len, size;
    long l, b = 0, n, s = 1;
    struct sort_data d;

    rb_scan_args(argc, argv, "03", &beg, &len, &size);
    l = RSTRING_LEN(str);
    if (!NIL_P(beg) && (b = NUM2INT(beg)) < 0 && (b += l) < 0) {
	rb_raise(rb_eArgError, "out of bounds");
    }
    if (!NIL_P(size) && (s = NUM2INT(size)) < 0) {
	rb_raise(rb_eArgError, "negative size");
    }
    if (NIL_P(len) ||
	(((n = NUM2INT(len)) < 0) ?
	 (rb_raise(rb_eArgError, "negative length"), 0) :
	 (b + n * s > l))) {
	n = (l - b) / s;
    }
    rb_str_modify(str);
    d.enc = rb_enc_get(str);
    d.elsize = s;
    ruby_qsort(RSTRING_PTR(str) + b, n, s,
	       rb_block_given_p() ? cmp_1 : cmp_2, &d);
    return str;
}

void
Init_string_qsort(VALUE klass)
{
    rb_define_method(klass, "qsort!", bug_str_qsort_bang, -1);
}
