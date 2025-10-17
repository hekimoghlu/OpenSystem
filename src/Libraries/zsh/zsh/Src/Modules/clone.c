/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
 * The clone builtin can be used to start a forked instance of the current
 * shell on a new terminal.  The only argument to the builtin is the name
 * of the new terminal.  In the new shell the PID, PPID and TTY parameters
 * are changed appropriately.  $! is set to zero in the new instance of the
 * shell and to the pid of the new instance in the original shell.
 *
 */

#include "clone.mdh"
#include "clone.pro"

/**/
static int
bin_clone(char *nam, char **args, UNUSED(Options ops), UNUSED(int func))
{
    int ttyfd, pid, cttyfd;

    unmetafy(*args, NULL);
    ttyfd = open(*args, O_RDWR|O_NOCTTY);
    if (ttyfd < 0) {
	zwarnnam(nam, "%s: %e", *args, errno);
	return 1;
    }
    pid = fork();
    if (!pid) {
	clearjobtab(0);
	ppid = getppid();
	mypid = getpid();
#ifdef HAVE_SETSID
	if (setsid() != mypid)
	    zwarnnam(nam, "failed to create new session: %e", errno);
#elif defined(TIOCNOTTY)
	    if (ioctl(SHTTY, TIOCNOTTY, 0))
	    zwarnnam(*args, "%e", errno);
	    setpgrp(0L, mypid);
#endif
	dup2(ttyfd,0);
	dup2(ttyfd,1);
	dup2(ttyfd,2);
	if (ttyfd > 2)
	    close(ttyfd);
	closem(FDT_UNUSED, 0);
	close(coprocin);
	close(coprocout);
	/* Acquire a controlling terminal */
	cttyfd = open(*args, O_RDWR);
	if (cttyfd == -1)
	    zwarnnam(nam, "%e", errno);
	else {
#ifdef TIOCSCTTY
	    ioctl(cttyfd, TIOCSCTTY, 0);
#endif
	    close(cttyfd);
	}
	/* check if we acquired the tty successfully */
	cttyfd = open("/dev/tty", O_RDWR);
	if (cttyfd == -1)
	    zwarnnam(nam, "could not make %s my controlling tty, job control "
		     "disabled", *args);
	else
	    close(cttyfd);

	/* Clear mygrp so that acquire_pgrp() gets the new process group.
	 * (acquire_pgrp() is called from init_io()) */
	mypgrp = 0;
	init_io(NULL);
	setsparam("TTY", ztrdup(ttystrname));
    }
    else
	close(ttyfd);
    if (pid < 0) {
	zerrnam(nam, "fork failed: %e", errno);
	return 1;
    }
    lastpid = pid;
    return 0;
}

static struct builtin bintab[] = {
    BUILTIN("clone", 0, bin_clone, 1, 1, 0, NULL, NULL),
};

static struct features module_features = {
    bintab, sizeof(bintab)/sizeof(*bintab),
    NULL, 0,
    NULL, 0,
    NULL, 0,
    0
};

/**/
int
setup_(UNUSED(Module m))
{
    return 0;
}

/**/
int
features_(Module m, char ***features)
{
    *features = featuresarray(m, &module_features);
    return 0;
}

/**/
int
enables_(Module m, int **enables)
{
    return handlefeatures(m, &module_features, enables);
}

/**/
int
boot_(UNUSED(Module m))
{
    return 0;
}

/**/
int
cleanup_(Module m)
{
    return setfeatureenables(m, &module_features, NULL);
}

/**/
int
finish_(UNUSED(Module m))
{
    return 0;
}
