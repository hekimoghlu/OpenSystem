/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

#include <ruby/ruby.h>
#include <ruby/vm.h>

static void
do_nothing(ruby_vm_t *vm)
{
}

static void
print_begin(ruby_vm_t *vm)
{
    printf("begin\n");
}

static void
print_end(ruby_vm_t *vm)
{
    printf("end\n");
}

static VALUE
register_at_exit(VALUE self, VALUE t)
{
    switch (t) {
      case Qtrue:
	ruby_vm_at_exit(print_begin);
	break;
      case Qfalse:
	ruby_vm_at_exit(print_end);
	break;
      default:
	ruby_vm_at_exit(do_nothing);
	break;
    }
    return self;
}

void
Init_at_exit(void)
{
    VALUE m = rb_define_module("Bug");
    VALUE c = rb_define_class_under(m, "VM", rb_cObject);
    rb_define_singleton_method(c, "register_at_exit", register_at_exit, 1);
}
