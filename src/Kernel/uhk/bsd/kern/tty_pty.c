/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
 * Copyright (c) 1982, 1986, 1989, 1993
 *      The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the University of
 *      California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)tty_pty.c	8.4 (Berkeley) 2/20/95
 */

/*
 * Pseudo-teletype Driver
 * (Actually two drivers, requiring two entries in 'cdevsw')
 */
#include "pty.h"                /* XXX */

#include <sys/param.h>
#include <sys/systm.h>
#include <sys/ioctl.h>
#include <sys/proc_internal.h>
#include <sys/kauth.h>
#include <sys/tty.h>
#include <sys/conf.h>
#include <sys/file_internal.h>
#include <sys/uio_internal.h>
#include <sys/kernel.h>
#include <sys/vnode.h>
#include <sys/user.h>
#include <sys/signalvar.h>

#if CONFIG_MACF
#include <security/mac_framework.h>
#endif

#include "tty_dev.h"

#if NPTY == 1
#undef NPTY
#define NPTY    32              /* crude XXX */
#warning        You have only one pty defined, redefining to 32.
#endif

/*
 * pts == /dev/tty[pqrsPQRS][0-9a-v]
 * ptc == /dev/pty[pqrsPQRS][0-9a-v]
 */
static struct ptmx_ioctl pt_ioctl[NPTY];

int pty_init(int n_ptys);

#ifndef DEVFS
int
pty_init(__unused int n_ptys)
{
	return 0;
}
#else // DEVFS
#include <miscfs/devfs/devfs.h>
#define START_CHAR      'p'
#define HEX_BASE        16

static struct tty_dev_t _pty_driver;

static struct ptmx_ioctl *
pty_get_ioctl(int minor, int open_flag)
{
	if (minor >= NPTY) {
		return NULL;
	}
	struct ptmx_ioctl *pti = &pt_ioctl[minor];
	if (open_flag & (PF_OPEN_M | PF_OPEN_S)) {
		if (!pti->pt_tty) {
			pti->pt_tty = ttymalloc();
		}
		if (!pti->pt_tty) {
			return NULL;
		}
	}
	return pti;
}

static int
pty_get_name(int minor, char *buffer, size_t size)
{
	return snprintf(buffer, size, "/dev/tty%c%x",
	           START_CHAR + (minor / HEX_BASE),
	           minor % HEX_BASE);
}

int
pty_init(int n_ptys)
{
	int i;
	int j;

	n_ptys = min(n_ptys, NPTY); /* clamp to avoid pt_ioctl overflow */

	/* create the pseudo tty device nodes */
	for (j = 0; j < 10; j++) {
		for (i = 0; i < HEX_BASE; i++) {
			int m = j * HEX_BASE + i;
			if (m >= n_ptys) {
				goto done;
			}
			pt_ioctl[m].pt_devhandle = devfs_make_node(makedev(PTS_MAJOR, m),
			    DEVFS_CHAR, UID_ROOT, GID_WHEEL, 0666,
			    "tty%c%x", j + START_CHAR, i);
			(void)devfs_make_node(makedev(PTC_MAJOR, m),
			    DEVFS_CHAR, UID_ROOT, GID_WHEEL, 0666,
			    "pty%c%x", j + START_CHAR, i);
		}
	}

done:
	_pty_driver.primary = PTC_MAJOR;
	_pty_driver.replica = PTS_MAJOR;
	_pty_driver.open_reset = 1;
	_pty_driver.open = &pty_get_ioctl;
	_pty_driver.name = &pty_get_name;
	tty_dev_register(&_pty_driver);

	if (cdevsw_setkqueueok(PTC_MAJOR, &cdevsw[PTC_MAJOR], CDEVSW_IS_PTC) == -1) {
		panic("Can't mark ptc as kqueue ok");
	}
	if (cdevsw_setkqueueok(PTS_MAJOR, &cdevsw[PTS_MAJOR], CDEVSW_IS_PTS) == -1) {
		panic("Can't mark pts as kqueue ok");
	}
	return 0;
}
#endif // DEVFS
