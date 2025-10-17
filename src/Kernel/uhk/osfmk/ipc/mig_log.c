/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
 * Revision 1.1.1.1  1998/09/22 21:05:29  wsanchez
 * Import of Mac OS X kernel (~semeria)
 *
 * Revision 1.1.1.1  1998/03/07 02:26:16  wsanchez
 * Import of OSF Mach kernel (~mburg)
 *
 * Revision 1.2.6.1  1994/09/23  02:14:23  ezf
 *      change marker to not FREE
 *      [1994/09/22  21:31:33  ezf]
 *
 * Revision 1.2.2.4  1993/08/03  18:29:18  gm
 *      CR9596: Change KERNEL to MACH_KERNEL.
 *      [1993/08/02  16:11:07  gm]
 *
 * Revision 1.2.2.3  1993/07/22  16:18:15  rod
 *      Add ANSI prototypes.  CR #9523.
 *      [1993/07/22  13:34:22  rod]
 *
 * Revision 1.2.2.2  1993/06/09  02:33:38  gm
 *      Added to OSF/1 R1.3 from NMK15.0.
 *      [1993/06/02  21:11:41  jeffc]
 *
 * Revision 1.2  1993/04/19  16:23:26  devrcs
 *      Untyped ipc merge:
 *      Support for logging and tracing within the MIG stubs
 *      [1993/02/24  14:49:29  travos]
 *
 * $EndLog$
 */

#ifdef MACH_KERNEL
#include <mig_debug.h>
#endif

#include <mach/message.h>
#include <mach/mig_log.h>

int mig_tracing, mig_errors, mig_full_tracing;

/*
 * Tracing facilities for MIG generated stubs.
 *
 * At the moment, there is only a printf, which is
 * activated through the runtime switch:
 *      mig_tracing to call MigEventTracer
 *      mig_errors to call MigEventErrors
 * For this to work, MIG has to run with the -L option,
 * and the mig_debug flags has to be selected
 *
 * In the future, it will be possible to collect infos
 * on the use of MACH IPC with an application similar
 * to netstat.
 *
 * A new option will be generated accordingly to the
 * kernel configuration rules, e.g
 *	#include <mig_log.h>
 */

void
MigEventTracer(
	mig_who_t               who,
	mig_which_event_t       what,
	mach_msg_id_t           msgh_id,
	unsigned int            size,
	unsigned int            kpd,
	unsigned int            retcode,
	unsigned int            ports,
	unsigned int            oolports,
	unsigned int            ool,
	char                    *file,
	unsigned int            line)
{
	printf("%d|%d|%d", who, what, msgh_id);
	if (mig_full_tracing) {
		printf(" -- sz%d|kpd%d|ret(0x%x)|p%d|o%d|op%d|%s, %d",
		    size, kpd, retcode, ports, oolports, ool, file, line);
	}
	printf("\n");
}

void
MigEventErrors(
	mig_who_t               who,
	mig_which_error_t       what,
	void                    *par,
	char                    *file,
	unsigned int            line)
{
	if (what == MACH_MSG_ERROR_UNKNOWN_ID) {
		printf("%d|%d|%d -- %s %d\n", who, what, *(int *)par, file, line);
	} else {
		printf("%d|%d|%s -- %s %d\n", who, what, (char *)par, file, line);
	}
}
