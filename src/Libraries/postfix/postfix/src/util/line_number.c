/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
  * System library.
  */
#include <sys_defs.h>

 /*
  * Utility library.
  */
#include <vstring.h>
#include <line_number.h>

/* format_line_number - pretty-print line number or number range */

char   *format_line_number(VSTRING *result, ssize_t first, ssize_t last)
{
    static VSTRING *buf;

    /*
     * Your buffer or mine?
     */
    if (result == 0) {
	if (buf == 0)
	    buf = vstring_alloc(10);
	result = buf;
    }

    /*
     * Print a range only when the numbers differ.
     */
    vstring_sprintf(result, "%ld", (long) first);
    if (first != last)
	vstring_sprintf_append(result, "-%ld", (long) last);

    return (vstring_str(result));
}
