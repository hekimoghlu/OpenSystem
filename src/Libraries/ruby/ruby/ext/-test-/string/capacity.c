/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#include "internal.h"

static VALUE
bug_str_capacity(VALUE klass, VALUE str)
{
    return
	STR_EMBED_P(str) ? INT2FIX(RSTRING_EMBED_LEN_MAX) : \
	STR_SHARED_P(str) ? INT2FIX(0) : \
	LONG2FIX(RSTRING(str)->as.heap.aux.capa);
}

void
Init_string_capacity(VALUE klass)
{
    rb_define_singleton_method(klass, "capacity", bug_str_capacity, 1);
}
