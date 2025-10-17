/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

#include "ruby/ruby.h"

static VALUE
arith_seq_s_extract(VALUE mod, VALUE obj)
{
  rb_arithmetic_sequence_components_t x;
  VALUE ret;
  int r;

  r = rb_arithmetic_sequence_extract(obj, &x);

  ret = rb_ary_new2(5);
  rb_ary_store(ret, 0, r ? x.begin : Qnil);
  rb_ary_store(ret, 1, r ? x.end   : Qnil);
  rb_ary_store(ret, 2, r ? x.step  : Qnil);
  rb_ary_store(ret, 3, r ? INT2FIX(x.exclude_end) : Qnil);
  rb_ary_store(ret, 4, INT2FIX(r));

  return ret;
}

void
Init_extract(void)
{
    VALUE cArithSeq = rb_path2class("Enumerator::ArithmeticSequence");
    rb_define_singleton_method(cArithSeq, "__extract__", arith_seq_s_extract, 1);
}
