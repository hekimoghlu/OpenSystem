/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
begin(VALUE object)
{
    return rb_funcall(object, rb_intern("try_method"), 0);
}

static VALUE
ensure(VALUE object)
{
    return rb_funcall(object, rb_intern("ensured_method"), 0);
}

static VALUE
ensured(VALUE module, VALUE object)
{
    return rb_ensure(begin, object, ensure, object);
}

static VALUE
exc_raise(VALUE exc)
{
    rb_exc_raise(exc);
    return Qnil;
}

static VALUE
ensure_raise(VALUE module, VALUE object, VALUE exc)
{
    return rb_ensure(rb_yield, object, exc_raise, exc);
}

void
Init_ensured(VALUE klass)
{
    rb_define_module_function(klass, "ensured", ensured, 1);
    rb_define_module_function(klass, "ensure_raise", ensure_raise, 2);
}
