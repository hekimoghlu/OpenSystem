/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef HAVE_PRIV_SET
# include <priv.h>
#endif
#include <errno.h>

#include "sudo.h"
#include "sudo_exec.h"

/*
 * Disable execution of child processes in the command we are about
 * to run.  On systems with privilege sets, we can remove the exec
 * privilege.  On other systems we use LD_PRELOAD and the like.
 */
char **
disable_execute(char *envp[], const char *dso)
{
    debug_decl(disable_execute, SUDO_DEBUG_UTIL);

#ifdef HAVE_PRIV_SET
    /* Solaris privileges, remove PRIV_PROC_EXEC post-execve. */
    (void)priv_set(PRIV_ON, PRIV_INHERITABLE, "PRIV_FILE_DAC_READ", NULL);
    (void)priv_set(PRIV_ON, PRIV_INHERITABLE, "PRIV_FILE_DAC_WRITE", NULL);
    (void)priv_set(PRIV_ON, PRIV_INHERITABLE, "PRIV_FILE_DAC_SEARCH", NULL);
    if (priv_set(PRIV_OFF, PRIV_LIMIT, "PRIV_PROC_EXEC", NULL) == 0)
	debug_return_ptr(envp);
    sudo_warn("%s", U_("unable to remove PRIV_PROC_EXEC from PRIV_LIMIT"));
#endif /* HAVE_PRIV_SET */

#ifdef RTLD_PRELOAD_VAR
    if (dso != NULL)
	envp = sudo_preload_dso(envp, dso, -1);
#endif /* RTLD_PRELOAD_VAR */

    debug_return_ptr(envp);
}

/*
 * Trap execution of child processes in the command we are about to run.
 * Uses LD_PRELOAD and the like to perform a policy check on child commands.
 */
static char **
enable_intercept(char *envp[], const char *dso, int intercept_fd)
{
    debug_decl(enable_intercept, SUDO_DEBUG_UTIL);

    if (dso != NULL) {
#ifdef RTLD_PRELOAD_VAR
	if (intercept_fd == -1)
	    sudo_fatalx("%s: no intercept fd", __func__);

	envp = sudo_preload_dso(envp, dso, intercept_fd);
#else
	/* Intercept not supported, envp unchanged. */
	if (intercept_fd != -1)
	    close(intercept_fd);
#endif /* RTLD_PRELOAD_VAR */
    }

    debug_return_ptr(envp);
}

/*
 * Like execve(2) but falls back to running through /bin/sh
 * ala execvp(3) if we get ENOEXEC.
 */
int
sudo_execve(int fd, const char *path, char *const argv[], char *envp[],
    int intercept_fd, int flags)
{
    debug_decl(sudo_execve, SUDO_DEBUG_UTIL);

    sudo_debug_execve(SUDO_DEBUG_INFO, path, argv, envp);

    /* Modify the environment as needed to trap execve(). */
    if (ISSET(flags, CD_NOEXEC))
	envp = disable_execute(envp, sudo_conf_noexec_path());
    if (ISSET(flags, CD_INTERCEPT|CD_LOG_SUBCMDS)) {
	if (!ISSET(flags, CD_USE_PTRACE)) {
	    envp = enable_intercept(envp, sudo_conf_intercept_path(),
		intercept_fd);
	}
    }

#ifdef HAVE_FEXECVE
    if (fd != -1)
	    fexecve(fd, argv, envp);
    else
#endif
	    execve(path, argv, envp);
    if (fd == -1 && errno == ENOEXEC) {
	int argc;
	const char **nargv;

	for (argc = 0; argv[argc] != NULL; argc++)
	    continue;
	nargv = reallocarray(NULL, argc + 2, sizeof(char *));
	if (nargv != NULL) {
	    nargv[0] = "sh";
	    nargv[1] = path;
	    memcpy(nargv + 2, argv + 1, argc * sizeof(char *));
	    execve(_PATH_SUDO_BSHELL, (char **)nargv, envp);
	    free(nargv);
	}
    }
    debug_return_int(-1);
}
