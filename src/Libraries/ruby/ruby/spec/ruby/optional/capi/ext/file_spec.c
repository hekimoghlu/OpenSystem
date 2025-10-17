/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#ifdef HAVE_RB_FILE_OPEN
VALUE file_spec_rb_file_open(VALUE self, VALUE name, VALUE mode) {
  return rb_file_open(RSTRING_PTR(name), RSTRING_PTR(mode));
}
#endif

#ifdef HAVE_RB_FILE_OPEN_STR
VALUE file_spec_rb_file_open_str(VALUE self, VALUE name, VALUE mode) {
  return rb_file_open_str(name, RSTRING_PTR(mode));
}
#endif

#ifdef HAVE_FILEPATHVALUE
VALUE file_spec_FilePathValue(VALUE self, VALUE obj) {
  return FilePathValue(obj);
}
#endif

void Init_file_spec(void) {
  VALUE cls = rb_define_class("CApiFileSpecs", rb_cObject);

#ifdef HAVE_RB_FILE_OPEN
  rb_define_method(cls, "rb_file_open", file_spec_rb_file_open, 2);
#endif

#ifdef HAVE_RB_FILE_OPEN_STR
  rb_define_method(cls, "rb_file_open_str", file_spec_rb_file_open_str, 2);
#endif

#ifdef HAVE_FILEPATHVALUE
  rb_define_method(cls, "FilePathValue", file_spec_FilePathValue, 1);
#endif
}

#ifdef __cplusplus
}
#endif
