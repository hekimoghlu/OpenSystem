/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stddef.h>

#ifdef __osf__
#include <stdio.h>
#include <sys/sysinfo.h>
#include <sys/proc.h>
#endif /* __osf__ */

#include "varattrs.h"
#include "machdep.h"

/*
 * On platforms where the CPU doesn't support unaligned loads, force
 * unaligned accesses to abort with SIGBUS, rather than being fixed
 * up (slowly) by the OS kernel; on those platforms, misaligned accesses
 * are bugs, and we want tcpdump to crash so that the bugs are reported.
 *
 * The only OS on which this is necessary is DEC OSF/1^W^WDigital
 * UNIX^W^WTru64 UNIX.
 */
int
abort_on_misalignment(char *ebuf _U_, size_t ebufsiz _U_)
{
#ifdef __osf__
	static int buf[2] = { SSIN_UACPROC, UAC_SIGBUS };

	if (setsysinfo(SSI_NVPAIRS, (caddr_t)buf, 1, 0, 0) < 0) {
		(void)snprintf(ebuf, ebufsiz, "setsysinfo: errno %d", errno);
		return (-1);
	}
#endif
	return (0);
}
