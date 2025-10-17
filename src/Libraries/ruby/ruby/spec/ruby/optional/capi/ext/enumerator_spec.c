/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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

#ifdef HAVE_RB_ENUMERATORIZE
VALUE enumerator_spec_rb_enumeratorize(int argc, VALUE *argv, VALUE self) {
  VALUE obj, meth, args;
  rb_scan_args(argc, argv, "2*", &obj, &meth, &args);
  return rb_enumeratorize(obj, meth, (int)RARRAY_LEN(args), RARRAY_PTR(args));
}
#endif

#ifdef HAVE_RB_ENUMERATORIZE_WITH_SIZE
VALUE enumerator_spec_size_fn(VALUE obj, VALUE args, VALUE anEnum) {
  return INT2NUM(7);
}

VALUE enumerator_spec_rb_enumeratorize_with_size(int argc, VALUE *argv, VALUE self) {
  VALUE obj, meth, args;
  rb_scan_args(argc, argv, "2*", &obj, &meth, &args);
  return rb_enumeratorize_with_size(obj, meth, (int)RARRAY_LEN(args), RARRAY_PTR(args), enumerator_spec_size_fn);
}
#endif

void Init_enumerator_spec(void) {
  VALUE cls;
  cls = rb_define_class("CApiEnumeratorSpecs", rb_cObject);

#ifdef HAVE_RB_ENUMERATORIZE
  rb_define_method(cls, "rb_enumeratorize", enumerator_spec_rb_enumeratorize, -1);
#endif
#ifdef HAVE_RB_ENUMERATORIZE_WITH_SIZE
  rb_define_method(cls, "rb_enumeratorize_with_size", enumerator_spec_rb_enumeratorize_with_size, -1);
#endif
}

#ifdef __cplusplus
}
#endif
