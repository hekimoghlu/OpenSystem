/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
#include <tests/ktest_internal.h>
#include <kern/misc_protos.h>
#include <kern/debug.h>

int vsnprintf(char *, size_t, const char *, va_list);

void
ktest_set_current_expr(const char * expr_fmt, ...)
{
	int ret;
	va_list args;

	va_start(args, expr_fmt);
	ret = vsnprintf(ktest_current_expr, KTEST_MAXLEN, expr_fmt, args);
	va_end(args);
}

void
ktest_set_current_var(const char * name, const char * value_fmt, ...)
{
	int ret;
	va_list args;

	if (ktest_current_var_index >= KTEST_MAXVARS) {
		panic("Internal ktest error");
	}

	strlcpy(ktest_current_var_names[ktest_current_var_index],
	    name,
	    KTEST_MAXLEN);

	va_start(args, value_fmt);
	ret = vsnprintf(ktest_current_var_values[ktest_current_var_index],
	    KTEST_MAXLEN,
	    value_fmt,
	    args);
	va_end(args);

	ktest_current_var_index++;
}

void
ktest_set_current_msg(const char * msg, ...)
{
	int ret;
	va_list args;

	if (msg == NULL) {
		return;
	}

	va_start(args, msg);
	ret = vsnprintf(ktest_current_msg, KTEST_MAXLEN, msg, args);
	va_end(args);
}
