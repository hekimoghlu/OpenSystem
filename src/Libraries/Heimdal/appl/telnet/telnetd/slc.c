/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
#include "telnetd.h"

RCSID("$Id$");

/*
 * get_slc_defaults
 *
 * Initialize the slc mapping table.
 */
void
get_slc_defaults(void)
{
    int i;

    init_termbuf();

    for (i = 1; i <= NSLC; i++) {
	slctab[i].defset.flag =
	    spcset(i, &slctab[i].defset.val, &slctab[i].sptr);
	slctab[i].current.flag = SLC_NOSUPPORT;
	slctab[i].current.val = 0;
    }

}
