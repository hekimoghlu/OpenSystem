/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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

#ifdef HAVE_RB_RANGE_NEW
VALUE range_spec_rb_range_new(int argc, VALUE* argv, VALUE self) {
  int exclude_end = 0;
  if(argc == 3) {
    exclude_end = RTEST(argv[2]);
  }
  return rb_range_new(argv[0], argv[1], exclude_end);
}
#endif

#ifdef HAVE_RB_RANGE_VALUES
VALUE range_spec_rb_range_values(VALUE self, VALUE range) {
  VALUE beg;
  VALUE end;
  int excl;
  VALUE ary = rb_ary_new();
  rb_range_values(range, &beg, &end, &excl);
  rb_ary_store(ary, 0, beg);
  rb_ary_store(ary, 1, end);
  rb_ary_store(ary, 2, excl ? Qtrue : Qfalse);
  return ary;
}
#endif

#ifdef HAVE_RB_RANGE_BEG_LEN
VALUE range_spec_rb_range_beg_len(VALUE self, VALUE range, VALUE begpv, VALUE lenpv, VALUE lenv, VALUE errv) {
  long begp = FIX2LONG(begpv);
  long lenp = FIX2LONG(lenpv);
  long len = FIX2LONG(lenv);
  int err = FIX2INT(errv);
  VALUE ary = rb_ary_new();
  VALUE res = rb_range_beg_len(range, &begp, &lenp, len, err);
  rb_ary_store(ary, 0, LONG2FIX(begp));
  rb_ary_store(ary, 1, LONG2FIX(lenp));
  rb_ary_store(ary, 2, res);
  return ary;
}
#endif

void Init_range_spec(void) {
  VALUE cls;
  cls = rb_define_class("CApiRangeSpecs", rb_cObject);

#ifdef HAVE_RB_RANGE_NEW
  rb_define_method(cls, "rb_range_new", range_spec_rb_range_new, -1);
#endif

#ifdef HAVE_RB_RANGE_VALUES
  rb_define_method(cls, "rb_range_values", range_spec_rb_range_values, 1);
#endif

#ifdef HAVE_RB_RANGE_BEG_LEN
  rb_define_method(cls, "rb_range_beg_len", range_spec_rb_range_beg_len, 5);
#endif
}

#ifdef __cplusplus
}
#endif
