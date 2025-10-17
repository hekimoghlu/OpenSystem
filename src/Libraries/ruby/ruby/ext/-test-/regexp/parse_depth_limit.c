/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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
#include <ruby/onigmo.h>

static VALUE
get_parse_depth_limit(VALUE self)
{
    unsigned int depth = onig_get_parse_depth_limit();
    return UINT2NUM(depth);
}

static VALUE
set_parse_depth_limit(VALUE self, VALUE depth)
{
    onig_set_parse_depth_limit(NUM2UINT(depth));
    return depth;
}

void
Init_parse_depth_limit(VALUE klass)
{
    rb_define_singleton_method(klass, "parse_depth_limit", get_parse_depth_limit, 0);
    rb_define_singleton_method(klass, "parse_depth_limit=", set_parse_depth_limit, 1);
}
