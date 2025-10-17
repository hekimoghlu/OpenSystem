/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
/* safe-level:
   0 - strings from streams/environment/ARGV are tainted (default)
   1 - no dangerous operation by tainted value
*/

#define SAFE_LEVEL_MAX RUBY_SAFE_LEVEL_MAX

#include "ruby/ruby.h"
#include "vm_core.h"

/* $SAFE accessor */

#undef rb_secure
#undef rb_set_safe_level
#undef ruby_safe_level_2_warning

int
ruby_safe_level_2_warning(void)
{
    return 2;
}

int
rb_safe_level(void)
{
    return GET_VM()->safe_level_;
}

void
rb_set_safe_level_force(int safe)
{
    GET_VM()->safe_level_ = safe;
}

void
rb_set_safe_level(int level)
{
    rb_vm_t *vm = GET_VM();

    if (level > SAFE_LEVEL_MAX) {
	rb_raise(rb_eArgError, "$SAFE=2 to 4 are obsolete");
    }
    else if (level < 0) {
	rb_raise(rb_eArgError, "$SAFE should be >= 0");
    }
    else {
	int line;
	const char *path = rb_source_location_cstr(&line);

	if (0) fprintf(stderr, "%s:%d $SAFE %d -> %d\n",
		       path ? path : "-", line, vm->safe_level_, level);

	vm->safe_level_ = level;
    }
}

static VALUE
safe_getter(void)
{
    return INT2NUM(rb_safe_level());
}

static void
safe_setter(VALUE val)
{
    int level = NUM2INT(val);
    rb_set_safe_level(level);
}

void
rb_secure(int level)
{
    if (level <= rb_safe_level()) {
	ID caller_name = rb_frame_callee();
	if (caller_name) {
	    rb_raise(rb_eSecurityError, "Insecure operation `%"PRIsVALUE"' at level %d",
		     rb_id2str(caller_name), rb_safe_level());
	}
	else {
	    rb_raise(rb_eSecurityError, "Insecure operation at level %d",
		     rb_safe_level());
	}
    }
}

void
rb_secure_update(VALUE obj)
{
}

void
rb_insecure_operation(void)
{
    ID caller_name = rb_frame_callee();
    if (caller_name) {
	rb_raise(rb_eSecurityError, "Insecure operation - %"PRIsVALUE,
		 rb_id2str(caller_name));
    }
    else {
	rb_raise(rb_eSecurityError, "Insecure operation: -r");
    }
}

void
rb_check_safe_obj(VALUE x)
{
    if (rb_safe_level() > 0 && OBJ_TAINTED(x)) {
	rb_insecure_operation();
    }
}

void
Init_safe(void)
{
    rb_define_virtual_variable("$SAFE", safe_getter, safe_setter);
}
