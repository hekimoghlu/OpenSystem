/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#include <sys/param.h>
#include "include/rpb.h"
#include "include/prom.h"


long alpha_rpcc(void);

int
getsecs()
{
	static long tnsec;
	static long lastpcc, wrapsecs;
	long curpcc;

	if (tnsec == 0) {
		tnsec = 1;
		lastpcc = alpha_rpcc() & 0xffffffff;
		wrapsecs = (0xffffffff /
		    ((struct rpb *)HWRPB_ADDR)->rpb_cc_freq) + 1;

#if 0
		printf("getsecs: cc freq = %d, time to wrap = %d\n",
		    ((struct rpb *)HWRPB_ADDR)->rpb_cc_freq, wrapsecs);
#endif
	}

	curpcc = alpha_rpcc() & 0xffffffff;
	if (curpcc < lastpcc)
		curpcc += 0x100000000;

	tnsec += ((curpcc - lastpcc) * 1000000000) / ((struct rpb *)HWRPB_ADDR)->rpb_cc_freq;
	lastpcc = curpcc;

	return (tnsec / 1000000000);
}

