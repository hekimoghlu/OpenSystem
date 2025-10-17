/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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

static VALUE boolean_spec_is_true(VALUE self, VALUE boolean) {
  if (boolean) {
    return INT2NUM(1);
  } else {
    return INT2NUM(2);
  }
}

static VALUE boolean_spec_q_true(VALUE self) {
  return Qtrue;
}

static VALUE boolean_spec_q_false(VALUE self) {
  return Qfalse;
}

void Init_boolean_spec(void) {
  VALUE cls;
  cls = rb_define_class("CApiBooleanSpecs", rb_cObject);
  rb_define_method(cls, "is_true", boolean_spec_is_true, 1);
  rb_define_method(cls, "q_true", boolean_spec_q_true, 0);
  rb_define_method(cls, "q_false", boolean_spec_q_false, 0);
}

#ifdef __cplusplus
extern "C" {
#endif
