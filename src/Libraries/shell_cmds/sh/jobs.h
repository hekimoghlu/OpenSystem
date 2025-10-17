/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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
/* Mode argument to forkshell.  Don't change FORK_FG or FORK_BG. */
#define FORK_FG 0
#define FORK_BG 1
#define FORK_NOJOB 2

#include <signal.h>		/* for sig_atomic_t */

struct job;

enum {
	SHOWJOBS_DEFAULT,	/* job number, status, command */
	SHOWJOBS_VERBOSE,	/* job number, PID, status, command */
	SHOWJOBS_PIDS,		/* PID only */
	SHOWJOBS_PGIDS		/* PID of the group leader only */
};

extern int job_warning;		/* user was warned about stopped jobs */

void setjobctl(int);
void showjobs(int, int);
struct job *makejob(union node *, int);
pid_t forkshell(struct job *, union node *, int);
pid_t vforkexecshell(struct job *, char **, char **, const char *, int, int []);
int waitforjob(struct job *, int *);
int stoppedjobs(void);
int backgndpidset(void);
pid_t backgndpidval(void);
char *commandtext(union node *);

#if ! JOBS
#define setjobctl(on)	/* do nothing */
#endif
