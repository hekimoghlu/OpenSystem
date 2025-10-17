/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
/*
 * Cephes Math Library Release 2.1:  January, 1989
 * Copyright 1984, 1987, 1989 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

#include "mconf.h"

double round(double x)
{
    double y, r;

    /* Largest integer <= x */
    y = floor(x);

    /* Fractional part */
    r = x - y;

    /* Round up to nearest. */
    if (r > 0.5)
	goto rndup;

    /* Round to even */
    if (r == 0.5) {
	r = y - 2.0 * floor(0.5 * y);
	if (r == 1.0) {
	  rndup:
	    y += 1.0;
	}
    }

    /* Else round down. */
    return (y);
}
