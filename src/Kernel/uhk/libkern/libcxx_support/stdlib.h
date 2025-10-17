/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#ifndef __STDLIB_H__
#define __STDLIB_H__

#include <sys/_types/_size_t.h>
#include <machine/trap.h>

typedef struct {
	int quot;
	int rem;
} div_t;

typedef struct {
	long quot;
	long rem;
} ldiv_t;

typedef struct {
	long long quot;
	long long rem;
} lldiv_t;

static inline div_t
div(int numer, int denom)
{
	div_t retval;

	retval.quot = numer / denom;
	retval.rem = numer % denom;
	if (numer >= 0 && retval.rem < 0) {
		retval.quot++;
		retval.rem -= denom;
	}
	return retval;
}

static inline ldiv_t
ldiv(long numer, long denom)
{
	ldiv_t retval;

	retval.quot = numer / denom;
	retval.rem = numer % denom;
	if (numer >= 0 && retval.rem < 0) {
		retval.quot++;
		retval.rem -= denom;
	}
	return retval;
}

static inline lldiv_t
lldiv(long long numer, long long denom)
{
	lldiv_t retval;

	retval.quot = numer / denom;
	retval.rem = numer % denom;
	if (numer >= 0 && retval.rem < 0) {
		retval.quot++;
		retval.rem -= denom;
	}
	return retval;
}

static inline void __attribute__((noreturn, cold))
abort(void)
{
	ml_fatal_trap(0x0800);
}

#endif
