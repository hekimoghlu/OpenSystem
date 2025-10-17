/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_util.h"
#include "sudo_fatal.h"
#include "check.h"

sudo_dso_public int main(int argc, char *argv[]);

#ifdef __linux__
static int
get_now(struct timespec *now)
{
    const char *errstr;
    char buf[1024];
    time_t seconds;
    int ret = -1;
    FILE *fp;

    /* Linux process start time is relative to boot time. */
    fp = fopen("/proc/stat", "r");
    if (fp != NULL) {
	while (fgets(buf, sizeof(buf), fp) != NULL) {
	    if (strncmp(buf, "btime ", 6) != 0)
		continue;
	    buf[strcspn(buf, "\n")] = '\0';

	    /* Boot time is in seconds since the epoch. */
	    seconds = sudo_strtonum(buf + 6, 0, TIME_T_MAX, &errstr);
	    if (errstr != NULL)
		return -1;

	    /* Instead of the real time, "now" is relative to boot time. */
	    if (sudo_gettime_real(now) == -1)
		return -1;
	    now->tv_sec -= seconds;
	    ret = 0;
	    break;
	}
	fclose(fp);
    }
    return ret;
}
#else
static int
get_now(struct timespec *now)
{
    /* Process start time is relative to wall clock time. */
    return sudo_gettime_real(now);
}
#endif

int
main(int argc, char *argv[])
{
    int ntests = 0, errors = 0;
    struct timespec now, then, delta;
    time_t timeoff = 0;
    pid_t pids[2];
    char *faketime;
    int i;

    initprogname(argc > 0 ? argv[0] : "check_starttime");

    if (get_now(&now) == -1)
	sudo_fatal_nodebug("unable to get current time");

    pids[0] = getpid();
    pids[1] = getppid();

    /* Debian CI pipeline runs tests using faketime. */
    faketime = getenv("FAKETIME");
    if (faketime != NULL)
	timeoff = sudo_strtonum(faketime, TIME_T_MIN, TIME_T_MAX, NULL);

    for (i = 0; i < 2; i++) {
	ntests++;
	if (get_starttime(pids[i], &then)  == -1) {
	    printf("%s: test %d: unable to get start time for pid %d\n",
		getprogname(), ntests, (int)pids[i]);
	    errors++;
	}
	if (i != 0)
	    continue;

	/* Verify our own process start time, allowing for some drift. */
	ntests++;
	sudo_timespecsub(&then, &now, &delta);
	delta.tv_sec += timeoff;
	if (delta.tv_sec > 30 || delta.tv_sec < -30) {
	    printf("%s: test %d: unexpected start time for pid %d: %s",
		getprogname(), ntests, (int)pids[i], ctime(&then.tv_sec));
	    errors++;
	}
    }

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }

    exit(errors);
}
