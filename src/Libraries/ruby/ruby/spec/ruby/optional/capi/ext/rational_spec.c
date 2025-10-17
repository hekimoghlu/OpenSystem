/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#include "rubyspec.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_RB_RATIONAL
static VALUE rational_spec_rb_Rational(VALUE self, VALUE num, VALUE den) {
  return rb_Rational(num, den);
}
#endif

#ifdef HAVE_RB_RATIONAL1
static VALUE rational_spec_rb_Rational1(VALUE self, VALUE num) {
  return rb_Rational1(num);
}
#endif

#ifdef HAVE_RB_RATIONAL2
static VALUE rational_spec_rb_Rational2(VALUE self, VALUE num, VALUE den) {
  return rb_Rational2(num, den);
}
#endif

#ifdef HAVE_RB_RATIONAL_NEW
static VALUE rational_spec_rb_rational_new(VALUE self, VALUE num, VALUE den) {
  return rb_rational_new(num, den);
}
#endif

#ifdef HAVE_RB_RATIONAL_NEW1
static VALUE rational_spec_rb_rational_new1(VALUE self, VALUE num) {
  return rb_rational_new1(num);
}
#endif

#ifdef HAVE_RB_RATIONAL_NEW2
static VALUE rational_spec_rb_rational_new2(VALUE self, VALUE num, VALUE den) {
  return rb_rational_new2(num, den);
}
#endif

#ifdef HAVE_RB_RATIONAL_NUM
static VALUE rational_spec_rb_rational_num(VALUE self, VALUE rational) {
  return rb_rational_num(rational);
}
#endif

#ifdef HAVE_RB_RATIONAL_DEN
static VALUE rational_spec_rb_rational_den(VALUE self, VALUE rational) {
  return rb_rational_den(rational);
}
#endif

void Init_rational_spec(void) {
  VALUE cls;
  cls = rb_define_class("CApiRationalSpecs", rb_cObject);

#ifdef HAVE_RB_RATIONAL
  rb_define_method(cls, "rb_Rational", rational_spec_rb_Rational, 2);
#endif

#ifdef HAVE_RB_RATIONAL1
  rb_define_method(cls, "rb_Rational1", rational_spec_rb_Rational1, 1);
#endif

#ifdef HAVE_RB_RATIONAL2
  rb_define_method(cls, "rb_Rational2", rational_spec_rb_Rational2, 2);
#endif

#ifdef HAVE_RB_RATIONAL_NEW
  rb_define_method(cls, "rb_rational_new", rational_spec_rb_rational_new, 2);
#endif

#ifdef HAVE_RB_RATIONAL_NEW1
  rb_define_method(cls, "rb_rational_new1", rational_spec_rb_rational_new1, 1);
#endif

#ifdef HAVE_RB_RATIONAL_NEW2
  rb_define_method(cls, "rb_rational_new2", rational_spec_rb_rational_new2, 2);
#endif

#ifdef HAVE_RB_RATIONAL_NUM
  rb_define_method(cls, "rb_rational_num", rational_spec_rb_rational_num, 1);
#endif

#ifdef HAVE_RB_RATIONAL_DEN
  rb_define_method(cls, "rb_rational_den", rational_spec_rb_rational_den, 1);
#endif
}

#ifdef __cplusplus
}
#endif
