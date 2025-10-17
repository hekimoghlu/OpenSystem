/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <sys/stat.h>
#include <sys/ioctl.h>
#ifdef HAVE_SYS_STROPTS_H
#include <sys/stropts.h>
#endif /* HAVE_SYS_STROPTS_H */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>

#if defined(HAVE_OPENPTY)
# if defined(HAVE_LIBUTIL_H)
#  include <libutil.h>		/* *BSD and macOS */
# endif
# if defined(HAVE_UTIL_H)
#  include <util.h>		/* macOS */
# elif defined(HAVE_PTY_H)
#  include <pty.h>		/* Linux */
# else
#  include <termios.h>		/* Solaris */
# endif
#endif

#include "sudo.h"

#if defined(HAVE_OPENPTY)
bool
get_pty(int *leader, int *follower, char *name, size_t namesz, uid_t ttyuid)
{
    struct group *gr;
    gid_t ttygid = (gid_t)-1;
    bool ret = false;
    debug_decl(get_pty, SUDO_DEBUG_PTY);

    if ((gr = getgrnam("tty")) != NULL)
	ttygid = gr->gr_gid;

    if (openpty(leader, follower, name, NULL, NULL) == 0) {
	if (chown(name, ttyuid, ttygid) == 0)
	    ret = true;
    }

    debug_return_bool(ret);
}

#elif defined(HAVE__GETPTY)
bool
get_pty(int *leader, int *follower, char *name, size_t namesz, uid_t ttyuid)
{
    char *line;
    bool ret = false;
    debug_decl(get_pty, SUDO_DEBUG_PTY);

    /* IRIX-style dynamic ptys (may fork) */
    line = _getpty(leader, O_RDWR, S_IRUSR|S_IWUSR|S_IWGRP, 0);
    if (line != NULL) {
	*follower = open(line, O_RDWR|O_NOCTTY, 0);
	if (*follower != -1) {
	    (void) chown(line, ttyuid, -1);
	    strlcpy(name, line, namesz);
	    ret = true;
	} else {
	    close(*leader);
	    *leader = -1;
	}
    }
    debug_return_bool(ret);
}
#elif defined(HAVE_GRANTPT)
# ifndef HAVE_POSIX_OPENPT
static int
posix_openpt(int oflag)
{
    int fd;

#  ifdef _AIX
    fd = open(_PATH_DEV "ptc", oflag);
#  else
    fd = open(_PATH_DEV "ptmx", oflag);
#  endif
    return fd;
}
# endif /* HAVE_POSIX_OPENPT */

bool
get_pty(int *leader, int *follower, char *name, size_t namesz, uid_t ttyuid)
{
    char *line;
    bool ret = false;
    debug_decl(get_pty, SUDO_DEBUG_PTY);

    *leader = posix_openpt(O_RDWR|O_NOCTTY);
    if (*leader != -1) {
	(void) grantpt(*leader); /* may fork */
	if (unlockpt(*leader) != 0) {
	    close(*leader);
	    goto done;
	}
	line = ptsname(*leader);
	if (line == NULL) {
	    close(*leader);
	    goto done;
	}
	*follower = open(line, O_RDWR|O_NOCTTY, 0);
	if (*follower == -1) {
	    close(*leader);
	    goto done;
	}
# if defined(I_PUSH) && !defined(_AIX)
	ioctl(*follower, I_PUSH, "ptem");	/* pseudo tty emulation module */
	ioctl(*follower, I_PUSH, "ldterm");	/* line discipline module */
# endif
	(void) chown(line, ttyuid, -1);
	strlcpy(name, line, namesz);
	ret = true;
    }
done:
    debug_return_bool(ret);
}

#else /* Old-style BSD ptys */

static char line[] = _PATH_DEV "ptyXX";

bool
get_pty(int *leader, int *follower, char *name, size_t namesz, uid_t ttyuid)
{
    char *bank, *cp;
    struct group *gr;
    gid_t ttygid = -1;
    bool ret = false;
    debug_decl(get_pty, SUDO_DEBUG_PTY);

    if ((gr = getgrnam("tty")) != NULL)
	ttygid = gr->gr_gid;

    for (bank = "pqrs"; *bank != '\0'; bank++) {
	line[sizeof(_PATH_DEV "ptyX") - 2] = *bank;
	for (cp = "0123456789abcdef"; *cp != '\0'; cp++) {
	    line[sizeof(_PATH_DEV "ptyXX") - 2] = *cp;
	    *leader = open(line, O_RDWR|O_NOCTTY, 0);
	    if (*leader == -1) {
		if (errno == ENOENT)
		    goto done; /* out of ptys */
		continue; /* already in use */
	    }
	    line[sizeof(_PATH_DEV "p") - 2] = 't';
	    (void) chown(line, ttyuid, ttygid);
	    (void) chmod(line, S_IRUSR|S_IWUSR|S_IWGRP);
# ifdef HAVE_REVOKE
	    (void) revoke(line);
# endif
	    *follower = open(line, O_RDWR|O_NOCTTY, 0);
	    if (*follower != -1) {
		    strlcpy(name, line, namesz);
		    ret = true; /* success */
		    goto done;
	    }
	    (void) close(*leader);
	}
    }
done:
    debug_return_bool(ret);
}
#endif /* HAVE_OPENPTY */
