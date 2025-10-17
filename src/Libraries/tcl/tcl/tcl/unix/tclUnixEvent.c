/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
#include "tclInt.h"
#ifndef HAVE_COREFOUNDATION	/* Darwin/Mac OS X CoreFoundation notifier is
				 * in tclMacOSXNotify.c */

/*
 *----------------------------------------------------------------------
 *
 * Tcl_Sleep --
 *
 *	Delay execution for the specified number of milliseconds.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Time passes.
 *
 *----------------------------------------------------------------------
 */

void
Tcl_Sleep(
    int ms)			/* Number of milliseconds to sleep. */
{
    struct timeval delay;
    Tcl_Time before, after, vdelay;

    /*
     * The only trick here is that select appears to return early under some
     * conditions, so we have to check to make sure that the right amount of
     * time really has elapsed.  If it's too early, go back to sleep again.
     */

    Tcl_GetTime(&before);
    after = before;
    after.sec += ms/1000;
    after.usec += (ms%1000)*1000;
    if (after.usec > 1000000) {
	after.usec -= 1000000;
	after.sec += 1;
    }
    while (1) {
	/*
	 * TIP #233: Scale from virtual time to real-time for select.
	 */

	vdelay.sec  = after.sec  - before.sec;
	vdelay.usec = after.usec - before.usec;

	if (vdelay.usec < 0) {
	    vdelay.usec += 1000000;
	    vdelay.sec  -= 1;
	}

	if ((vdelay.sec != 0) || (vdelay.usec != 0)) {
	    (*tclScaleTimeProcPtr) (&vdelay, tclTimeClientData);
	}

	delay.tv_sec  = vdelay.sec;
	delay.tv_usec = vdelay.usec;

	/*
	 * Special note: must convert delay.tv_sec to int before comparing to
	 * zero, since delay.tv_usec is unsigned on some platforms.
	 */

	if ((((int) delay.tv_sec) < 0)
		|| ((delay.tv_usec == 0) && (delay.tv_sec == 0))) {
	    break;
	}
	(void) select(0, (SELECT_MASK *) 0, (SELECT_MASK *) 0,
		(SELECT_MASK *) 0, &delay);
	Tcl_GetTime(&before);
    }
}

#endif /* HAVE_COREFOUNDATION */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
