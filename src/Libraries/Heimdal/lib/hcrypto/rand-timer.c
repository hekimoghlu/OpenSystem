/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <rand.h>

#include <roken.h>

#include "randi.h"

#ifndef WIN32 /* don't bother with this on windows */

static volatile int counter;
static volatile unsigned char *gdata; /* Global data */
static volatile int igdata;	/* Index into global data */
static int gsize;

static
RETSIGTYPE
sigALRM(int sig)
{
    if (igdata < gsize)
	gdata[igdata++] ^= counter & 0xff;

#ifndef HAVE_SIGACTION
    signal(SIGALRM, sigALRM); /* Reinstall SysV signal handler */
#endif
    SIGRETURN(0);
}

#ifndef HAVE_SETITIMER
static void
pacemaker(struct timeval *tv)
{
    fd_set fds;
    pid_t pid;
    pid = getppid();
    while(1){
	FD_ZERO(&fds);
	FD_SET(0, &fds);
	select(1, &fds, NULL, NULL, tv);
	kill(pid, SIGALRM);
    }
}
#endif

#ifdef HAVE_SIGACTION
/* XXX ugly hack, should perhaps use function from roken */
static RETSIGTYPE
(*fake_signal(int sig, RETSIGTYPE (*f)(int)))(int)
{
    struct sigaction sa, osa;
    sa.sa_handler = f;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, &osa);
    return osa.sa_handler;
}
#define signal(S, F) fake_signal((S), (F))
#endif

#endif /* WIN32*/

/*
 *
 */

static void
timer_seed(const void *indata, int size)
{
}

static int
timer_bytes(unsigned char *outdata, int size)
{
#ifdef WIN32
    return 0;
#else /* WIN32 */
    struct itimerval tv, otv;
    RETSIGTYPE (*osa)(int);
    int i, j;
#ifndef HAVE_SETITIMER
    RETSIGTYPE (*ochld)(int);
    pid_t pid;
#endif

    gdata = outdata;
    gsize = size;
    igdata = 0;

    osa = signal(SIGALRM, sigALRM);

    /* Start timer */
    tv.it_value.tv_sec = 0;
    tv.it_value.tv_usec = 10 * 1000; /* 10 ms */
    tv.it_interval = tv.it_value;
#ifdef HAVE_SETITIMER
    setitimer(ITIMER_REAL, &tv, &otv);
#else
    ochld = signal(SIGCHLD, SIG_IGN);
    pid = fork();
    if(pid == -1){
	signal(SIGCHLD, ochld != SIG_ERR ? ochld : SIG_DFL);
	des_not_rand_data(data, size);
	return;
    }
    if(pid == 0)
	pacemaker(&tv.it_interval);
#endif

    for(i = 0; i < 4; i++) {
	for (igdata = 0; igdata < size;) /* igdata++ in sigALRM */
	    counter++;
	for (j = 0; j < size; j++) /* Only use 2 bits each lap */
	    gdata[j] = (gdata[j]>>2) | (gdata[j]<<6);
    }
#ifdef HAVE_SETITIMER
    setitimer(ITIMER_REAL, &otv, 0);
#else
    kill(pid, SIGKILL);
    while(waitpid(pid, NULL, 0) != pid);
    signal(SIGCHLD, ochld != SIG_ERR ? ochld : SIG_DFL);
#endif
    signal(SIGALRM, osa != SIG_ERR ? osa : SIG_DFL);

    return 1;
#endif
}

static void
timer_cleanup(void)
{
}

static void
timer_add(const void *indata, int size, double entropi)
{
}

static int
timer_pseudorand(unsigned char *outdata, int size)
{
    return timer_bytes(outdata, size);
}

static int
timer_status(void)
{
#ifdef WIN32
    return 0;
#else
    return 1;
#endif
}

const RAND_METHOD hc_rand_timer_method = {
    timer_seed,
    timer_bytes,
    timer_cleanup,
    timer_add,
    timer_pseudorand,
    timer_status
};

const RAND_METHOD *
RAND_timer_method(void)
{
    return &hc_rand_timer_method;
}
