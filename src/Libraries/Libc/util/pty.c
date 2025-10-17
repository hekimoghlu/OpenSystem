/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
 * Copyright (c) 1990, 1993, 1994
 *	The Regents of the University of California.  All rights reserved.
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
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
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
 */


#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <util.h>
#include <syslog.h>

int openpty(int *aprimary, int *areplica, char *name, struct termios *termp, struct winsize *winp)
{
	int primary, replica;
	char rname[128];
	int errno_saved;

	if ((primary = posix_openpt(O_RDWR|O_NOCTTY)) < 0)
		return -1;
	if (grantpt(primary) < 0 || unlockpt(primary) < 0
	    || ptsname_r(primary, rname, sizeof(rname)) == -1
	    || (replica = open(rname, O_RDWR|O_NOCTTY, 0)) < 0) {
		errno_saved = errno;
		(void) close(primary);
		errno = errno_saved;
		return -1;
	}
	*aprimary = primary;
	*areplica = replica;
	if (name)
		strcpy(name, rname);
	if (termp)
		(void) tcsetattr(replica, TCSAFLUSH, termp);
	if (winp)
		(void) ioctl(replica, TIOCSWINSZ, (char *)winp);
	return (0);
}

int
forkpty(int *aprimary, char *name, struct termios *termp, struct winsize *winp)
{
	int primary, replica, pid;
	int errno_saved;

	if (openpty(&primary, &replica, name, termp, winp) == -1)
		return (-1);
	switch (pid = fork()) {
	case -1:
		errno_saved = errno;
		(void) close(primary);
		(void) close(replica);
		errno = errno_saved;
		return (-1);
	case 0:
		/* 
		 * child
		 */
		(void) close(primary);
		/*
		 * 4300297: login_tty() may fail to set the controlling tty.
		 * Since we have already forked, the best we can do is to 
		 * dup the replica as if login_tty() succeeded.
		 */
		if (login_tty(replica) < 0) {
			syslog(LOG_ERR, "forkpty: login_tty could't make controlling tty");
			(void) dup2(replica, 0);
			(void) dup2(replica, 1);
			(void) dup2(replica, 2);
			if (replica > 2)
				(void) close(replica);
		}
		return (0);
	}
	/*
	 * parent
	 */
	*aprimary = primary;
	(void) close(replica);
	return (pid);
}
