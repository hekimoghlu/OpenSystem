/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include <darwintest.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(sigcontreturn, "checks that a call to waitid() for a child that is stopped and then continued returns correctly", T_META_TAG_VM_PREFERRED)
{
	pid_t           pid;
	siginfo_t       siginfo;
	pid = fork();
	T_QUIET; T_ASSERT_NE_INT(pid, -1, "fork() failed!");

	if (pid == 0) {
		while (1) {
		}
	}

	kill(pid, SIGSTOP);
	kill(pid, SIGCONT);
	sleep(1);

	T_QUIET; T_ASSERT_POSIX_SUCCESS(waitid(P_PID, pid, &siginfo, WCONTINUED), "Calling waitid() failed for pid %d", pid);

	T_ASSERT_EQ_INT(siginfo.si_status, SIGCONT, "A call to waitid() for stopped and continued child returns 0x%x, expected SIGCONT (0x%x)", siginfo.si_status, SIGCONT );
	kill(pid, SIGKILL);
}
