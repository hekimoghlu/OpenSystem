/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_RB_FLOAT_NEW
static VALUE float_spec_new_zero(VALUE self) {
  double flt = 0;
  return rb_float_new(flt);
}

static VALUE float_spec_new_point_five(VALUE self) {
  double flt = 0.555;
  return rb_float_new(flt);
}
#endif

#ifdef HAVE_RB_RFLOAT
static VALUE float_spec_rb_Float(VALUE self, VALUE float_str) {
  return rb_Float(float_str);
}
#endif

#ifdef HAVE_RFLOAT_VALUE
static VALUE float_spec_RFLOAT_VALUE(VALUE self, VALUE float_h) {
  return rb_float_new(RFLOAT_VALUE(float_h));
}
#endif

void Init_float_spec(void) {
  VALUE cls;
  cls = rb_define_class("CApiFloatSpecs", rb_cObject);

#ifdef HAVE_RB_FLOAT_NEW
  rb_define_method(cls, "new_zero", float_spec_new_zero, 0);
  rb_define_method(cls, "new_point_five", float_spec_new_point_five, 0);
#endif

#ifdef HAVE_RB_RFLOAT
  rb_define_method(cls, "rb_Float", float_spec_rb_Float, 1);
#endif

#ifdef HAVE_RFLOAT_VALUE
  rb_define_method(cls, "RFLOAT_VALUE", float_spec_RFLOAT_VALUE, 1);
#endif
}

#ifdef __cplusplus
}
#endif
