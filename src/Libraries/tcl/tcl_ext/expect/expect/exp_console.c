/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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
#include "expect_cf.h"
#include <stdio.h>
#include <sys/types.h>
#include <sys/ioctl.h>

/* Solaris needs this for console redir */
#ifdef HAVE_STRREDIR_H
#include <sys/strredir.h>
# ifdef SRIOCSREDIR
#  undef TIOCCONS
# endif
#endif

#ifdef HAVE_SYS_FCNTL_H
#include <sys/fcntl.h>
#endif

#include "tcl.h"
#include "exp_rename.h"
#include "exp_prog.h"
#include "exp_command.h"
#include "exp_log.h"

static void
exp_console_manipulation_failed(s)
char *s;
{
    expErrorLog("expect: spawn: cannot %s console, check permissions of /dev/console\n",s);
    exit(-1);
}

void
exp_console_set()
{
#ifdef SRIOCSREDIR
	int fd;

	if ((fd = open("/dev/console", O_RDONLY)) == -1) {
		exp_console_manipulation_failed("open");
	}
	if (ioctl(fd, SRIOCSREDIR, 0) == -1) {
		exp_console_manipulation_failed("redirect");
	}
	close(fd);
#endif

#ifdef TIOCCONS
	int on = 1;

	if (ioctl(0,TIOCCONS,(char *)&on) == -1) {
		exp_console_manipulation_failed("redirect");
	}
#endif /*TIOCCONS*/
}
