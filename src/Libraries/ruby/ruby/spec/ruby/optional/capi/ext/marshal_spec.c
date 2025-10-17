/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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

#ifdef HAVE_RB_MARSHAL_DUMP
VALUE marshal_spec_rb_marshal_dump(VALUE self, VALUE obj, VALUE port) {
  return rb_marshal_dump(obj, port);
}
#endif

#ifdef HAVE_RB_MARSHAL_LOAD
VALUE marshal_spec_rb_marshal_load(VALUE self, VALUE data) {
  return rb_marshal_load(data);
}
#endif

void Init_marshal_spec(void) {
  VALUE cls;
  cls = rb_define_class("CApiMarshalSpecs", rb_cObject);

#ifdef HAVE_RB_MARSHAL_DUMP
  rb_define_method(cls, "rb_marshal_dump", marshal_spec_rb_marshal_dump, 2);
#endif

#ifdef HAVE_RB_MARSHAL_LOAD
  rb_define_method(cls, "rb_marshal_load", marshal_spec_rb_marshal_load, 1);
#endif

}

#ifdef __cplusplus
}
#endif
