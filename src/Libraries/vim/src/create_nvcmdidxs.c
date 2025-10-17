/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
 * create_nvcmdidxs.c: helper program for `make nvcmdidxs`
 *
 * This outputs the list of command characters from the nv_cmds table in
 * decimal form, one per line.
 */

#include "vim.h"

// Declare nv_cmds[].
#include "nv_cmds.h"

#include <stdio.h>

int main(void)
{
    size_t i;

    for (i = 0; i < NV_CMDS_SIZE; i++)
    {
	int cmdchar = nv_cmds[i];

	// Special keys are negative, use the negated value for sorting.
	if (cmdchar < 0)
	    cmdchar = -cmdchar;
	printf("%d\n", cmdchar);
    }
    return 0;
}
