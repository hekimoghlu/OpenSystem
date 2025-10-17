/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

static VALUE
yield_block(int argc, VALUE *argv, VALUE self)
{
    rb_check_arity(argc, 1, UNLIMITED_ARGUMENTS);
    return rb_block_call(self, rb_to_id(argv[0]), argc-1, argv+1, rb_yield_block, 0);
}

void
Init_yield(VALUE klass)
{
    VALUE yield = rb_define_module_under(klass, "Yield");

    rb_define_method(yield, "yield_block", yield_block, -1);
}
