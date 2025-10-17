/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_RB_EXC_NEW
VALUE exception_spec_rb_exc_new(VALUE self, VALUE str) {
  char *cstr = StringValuePtr(str);
  return rb_exc_new(rb_eException, cstr, strlen(cstr));
}
#endif

#ifdef HAVE_RB_EXC_NEW2
VALUE exception_spec_rb_exc_new2(VALUE self, VALUE str) {
  char *cstr = StringValuePtr(str);
  return rb_exc_new2(rb_eException, cstr);
}
#endif

#ifdef HAVE_RB_EXC_NEW3
VALUE exception_spec_rb_exc_new3(VALUE self, VALUE str) {
  return rb_exc_new3(rb_eException, str);
}
#endif

#ifdef HAVE_RB_EXC_RAISE
VALUE exception_spec_rb_exc_raise(VALUE self, VALUE exc) {
    if (self != Qundef) rb_exc_raise(exc);
  return Qnil;
}
#endif

#ifdef HAVE_RB_SET_ERRINFO
VALUE exception_spec_rb_set_errinfo(VALUE self, VALUE exc) {
  rb_set_errinfo(exc);
  return Qnil;
}
#endif

void Init_exception_spec(void) {
  VALUE cls;
  cls = rb_define_class("CApiExceptionSpecs", rb_cObject);

#ifdef HAVE_RB_EXC_NEW
  rb_define_method(cls, "rb_exc_new", exception_spec_rb_exc_new, 1);
#endif

#ifdef HAVE_RB_EXC_NEW2
  rb_define_method(cls, "rb_exc_new2", exception_spec_rb_exc_new2, 1);
#endif

#ifdef HAVE_RB_EXC_NEW3
  rb_define_method(cls, "rb_exc_new3", exception_spec_rb_exc_new3, 1);
#endif

#ifdef HAVE_RB_EXC_RAISE
  rb_define_method(cls, "rb_exc_raise", exception_spec_rb_exc_raise, 1);
#endif

#ifdef HAVE_RB_SET_ERRINFO
  rb_define_method(cls, "rb_set_errinfo", exception_spec_rb_set_errinfo, 1);
#endif
}

#ifdef __cplusplus
}
#endif
