/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>

#include "sudo.h"
#include "sudo_exec.h"

void
suspend_sudo_nopty(struct exec_closure *ec, int signo, pid_t my_pid,
    pid_t my_pgrp, pid_t cmnd_pid)
{
    struct sigaction sa, osa;
    pid_t saved_pgrp = -1;
    int fd;
    debug_decl(suspend_sudo_nopty, SUDO_DEBUG_EXEC);

    /*
     * Save the controlling terminal's process group so we can restore
     * it after we resume, if needed.  Most well-behaved shells change
     * the pgrp back to its original value before suspending so we must
     * not try to restore in that case, lest we race with the command
     * upon resume, potentially stopping sudo with SIGTTOU while the
     * command continues to run.
     */
    fd = open(_PATH_TTY, O_RDWR);
    if (fd != -1) {
	saved_pgrp = tcgetpgrp(fd);
	if (saved_pgrp == -1) {
	    close(fd);
	    fd = -1;
	}
    }

    if (saved_pgrp != -1) {
	/*
	 * Command was stopped trying to access the controlling
	 * terminal.  If the command has a different pgrp and we
	 * own the controlling terminal, give it to the command's
	 * pgrp and let it continue.
	 */
	if (signo == SIGTTOU || signo == SIGTTIN) {
	    if (saved_pgrp == my_pgrp) {
		pid_t cmnd_pgrp = getpgid(cmnd_pid);
		if (cmnd_pgrp != my_pgrp) {
		    if (tcsetpgrp_nobg(fd, cmnd_pgrp) == 0) {
			if (killpg(cmnd_pgrp, SIGCONT) != 0)
			    sudo_warn("kill(%d, SIGCONT)", (int)cmnd_pgrp);
			close(fd);
			debug_return;
		    }
		}
	    }
	}
    }

    /* Log the suspend event. */
    log_suspend(ec, signo);

    if (signo == SIGTSTP) {
	memset(&sa, 0, sizeof(sa));
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART;
	sa.sa_handler = SIG_DFL;
	if (sudo_sigaction(SIGTSTP, &sa, &osa) != 0)
	    sudo_warn(U_("unable to set handler for signal %d"), SIGTSTP);
    }
    if (kill(my_pid, signo) != 0)
	sudo_warn("kill(%d, %d)", (int)my_pid, signo);
    if (signo == SIGTSTP) {
	if (sudo_sigaction(SIGTSTP, &osa, NULL) != 0)
	    sudo_warn(U_("unable to restore handler for signal %d"), SIGTSTP);
    }

    /* Log the resume event. */
    log_suspend(ec, SIGCONT);

    if (saved_pgrp != -1) {
	/*
	 * On resume, restore foreground process group, if different.
	 * Otherwise, we cannot resume some shells (pdksh).
	 *
	 * It is possible that we are no longer the foreground process,
	 * use tcsetpgrp_nobg() to prevent sudo from receiving SIGTTOU.
	 */
	if (saved_pgrp != my_pgrp)
	    tcsetpgrp_nobg(fd, saved_pgrp);
	close(fd);
    }

    debug_return;
}
