/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#define init(n) {void Init_##n(VALUE klass); Init_##n(klass);}

static VALUE
sym_find(VALUE dummy, VALUE sym)
{
    return rb_check_symbol(&sym);
}

static VALUE
sym_pinneddown_p(VALUE dummy, VALUE sym)
{
    ID id = rb_check_id(&sym);
    if (!id) return Qnil;
#ifdef ULL2NUM
    return ULL2NUM(id);
#else
    return ULONG2NUM(id);
#endif
}

void
Init_symbol(void)
{
    VALUE mBug = rb_define_module("Bug");
    VALUE klass = rb_define_class_under(mBug, "Symbol", rb_cSymbol);
    rb_define_singleton_method(klass, "find", sym_find, 1);
    rb_define_singleton_method(klass, "pinneddown?", sym_pinneddown_p, 1);
    TEST_INIT_FUNCS(init);
}
