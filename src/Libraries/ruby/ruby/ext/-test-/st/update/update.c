/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#include <ruby.h>
#include <ruby/st.h>

static int
update_func(st_data_t *key, st_data_t *value, st_data_t arg, int existing)
{
    VALUE ret = rb_yield_values(existing ? 2 : 1, (VALUE)*key, (VALUE)*value);
    switch (ret) {
      case Qfalse:
	return ST_STOP;
      case Qnil:
	return ST_DELETE;
      default:
	*value = ret;
	return ST_CONTINUE;
    }
}

static VALUE
test_st_update(VALUE self, VALUE key)
{
    if (st_update(RHASH_TBL(self), (st_data_t)key, update_func, 0))
	return Qtrue;
    else
	return Qfalse;
}

void
Init_update(void)
{
    VALUE st = rb_define_class_under(rb_define_module("Bug"), "StTable", rb_cHash);
    rb_define_method(st, "st_update", test_st_update, 1);
}

