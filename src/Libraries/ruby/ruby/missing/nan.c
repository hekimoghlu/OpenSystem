/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

#include "ruby/missing.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

double
nan(const char *spec)
{
#if 0
    /* FIXME: we have not yet seen any situation this is
     * necessary. Please write a proper implementation that
     * covers this branch.  */
    if (spec && spec[0]) {
	double generated_nan;
	int len = snprintf(NULL, 0, "NAN(%s)", spec);
	char *buf = malloc(len + 1); /* +1 for NUL */
	sprintf(buf, "NAN(%s)", spec);
	generated_nan = strtod(buf, NULL);
	free(buf);
	return generated_nan;
    }
    else
#endif
    {
	assert(!spec || !spec[0]);
	return (double)NAN;
    }
}
