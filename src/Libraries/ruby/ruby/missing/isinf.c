/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
#ifdef __osf__

#define _IEEE 1
#include <nan.h>

int
isinf(double n)
{
    if (IsNANorINF(n) && IsINF(n)) {
	return 1;
    }
    else {
	return 0;
    }
}

#else

#include "ruby/config.h"

#if defined(HAVE_FINITE) && defined(HAVE_ISNAN)

#include <math.h>
#ifdef HAVE_IEEEFP_H
#include <ieeefp.h>
#endif

/*
 * isinf may be provided only as a macro.
 * ex. HP-UX, Solaris 10
 * http://www.gnu.org/software/automake/manual/autoconf/Function-Portability.html
 */
#ifndef isinf
int
isinf(double n)
{
    return (!finite(n) && !isnan(n));
}
#endif

#else

#ifdef HAVE_STRING_H
# include <string.h>
#else
# include <strings.h>
#endif

static double zero(void) { return 0.0; }
static double one (void) { return 1.0; }
static double inf (void) { return one() / zero(); }

int
isinf(double n)
{
    static double pinf = 0.0;
    static double ninf = 0.0;

    if (pinf == 0.0) {
	pinf = inf();
	ninf = -pinf;
    }
    return memcmp(&n, &pinf, sizeof n) == 0
	|| memcmp(&n, &ninf, sizeof n) == 0;
}
#endif
#endif
