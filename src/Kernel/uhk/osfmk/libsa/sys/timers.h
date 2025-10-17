/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
 * @OSF_COPYRIGHT@
 */
/*
 * HISTORY
 *
 * Revision 1.1.1.1  1998/09/22 21:05:51  wsanchez
 * Import of Mac OS X kernel (~semeria)
 *
 * Revision 1.1.1.1  1998/03/07 02:25:36  wsanchez
 * Import of OSF Mach kernel (~mburg)
 *
 * Revision 1.1.4.1  1997/01/31  15:46:34  emcmanus
 *      Merged with nmk22b1_shared.
 *      [1997/01/30  08:47:46  emcmanus]
 *
 * Revision 1.1.2.2  1996/11/29  13:04:58  emcmanus
 *      Added TIMEOFDAY and getclock() prototype.
 *      [1996/11/29  09:59:33  emcmanus]
 *
 * Revision 1.1.2.1  1996/10/14  13:31:49  emcmanus
 *      Created.
 *      [1996/10/14  13:30:09  emcmanus]
 *
 * $EndLog$
 */

#ifndef _SYS_TIMERS_H_
#define _SYS_TIMERS_H_

/* POSIX <sys/timers.h>.  For now, we define just enough to be able to build
 *  the pthread library, with its pthread_cond_timedwait() interface.  */
struct timespec {
	unsigned long tv_sec;
	long tv_nsec;
};

#define TIMEOFDAY 1

extern int getclock(int, struct timespec *);

#endif  /* _SYS_TIMERS_H_ */
