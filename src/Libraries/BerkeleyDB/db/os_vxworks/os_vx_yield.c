/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#include "db_config.h"

#include "db_int.h"

/*
 * __os_yield --
 *	Yield the processor, optionally pausing until running again.
 */
void
__os_yield(env, secs, usecs)
	ENV *env;
	u_long secs, usecs;		/* Seconds and microseconds. */
{
	int ticks_delay, ticks_per_second;

	COMPQUIET(env, NULL);

	/* Don't require the values be normalized. */
	for (; usecs >= US_PER_SEC; usecs -= US_PER_SEC)
		++secs;

	/*
	 * Yield the processor so other processes or threads can run.
	 *
	 * As a side effect, taskDelay() moves the calling task to the end of
	 * the ready queue for tasks of the same priority. In particular, you
	 * can yield the CPU to any other tasks of the same priority by
	 * "delaying" for zero clock ticks.
	 *
	 * Never wait less than a tick, if we were supposed to wait at all.
	 */
	ticks_per_second = sysClkRateGet();
	ticks_delay =
	    secs * ticks_per_second + (usecs * ticks_per_second) / US_PER_SEC;
	if (ticks_delay == 0 && (secs != 0 || usecs != 0))
		ticks_delay = 1;
	(void)taskDelay(ticks_delay);
}
