/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#include "fork_program.h"
#include "kext_tools_util.h"
#include <spawn.h>
#include <sys/wait.h>
#include <libc.h>
#include <crt_externs.h>

/*******************************************************************************
* Fork a process after a specified delay, and either wait on it to exit or
* leave it to run in the background.
*
* Returns -2 on spawn() failure, -1 on other failure, and depending on wait:
* wait: true - exit status of forked program
* wait: false - pid of background process
*******************************************************************************/
int fork_program(const char * argv0, char * const argv[], Boolean wait)
{
    int            result;
    int            spawn_result;
    pid_t          child_pid;
    int            child_status;
    int            normal_iopolicy = getiopolicy_np(IOPOL_TYPE_DISK,
                                                    IOPOL_SCOPE_PROCESS);
    char ** environ = *(_NSGetEnviron());

#if 0 // spew program and arguments we are forking...
    if (argv0) {
        int i;
        int commandLen = 0;

        OSKextLog(NULL,
                  kOSKextLogErrorLevel | kOSKextLogGeneralFlag,
                  "Forking: %s",
                  argv0);
        for (i = 0; argv[i] != NULL; i++) {
            commandLen += strlen(argv[i]);
            commandLen++;
        }
        if (commandLen > 0) {
            char * myCmd = NULL;
            myCmd = (char *) malloc(commandLen);
            if (myCmd) {
                for (i = 0; argv[i] != NULL; i++) {
                    strcat(myCmd, argv[i]);
                    strcat(myCmd, " ");
                }
                OSKextLog(NULL,
                          kOSKextLogErrorLevel | kOSKextLogGeneralFlag,
                          "%s ",
                          myCmd);
                free(myCmd);
            }
        }
    }
#endif

    if (!wait) {
        setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, IOPOL_UTILITY);
    }

    spawn_result = posix_spawn(&child_pid, argv0, /* file_actions */ NULL,
        /* spawnattrs */ NULL, argv, environ);

    // If we couldn't spawn the process, return -2 with errno for detail
    if (spawn_result != 0) {
        OSKextLog(/* kext */ NULL, kOSKextLogErrorLevel,
            "posix_spawn failed for %s.", argv0);
        errno = spawn_result;
        result = -2;
        goto finish;
    }

    OSKextLog(/* kext */ NULL, kOSKextLogDetailLevel,
              "started child process %s[%d] (%ssynchronous).",
              argv0, child_pid, wait ? "" : "a");

    if (wait) {
        OSKextLogSpec logSpec = kOSKextLogDetailLevel;
        if (waitpid(child_pid, &child_status, 0) == -1) {
            result = -1;
            goto finish;
        }
        if (WIFEXITED(child_status)) {
            result = WEXITSTATUS(child_status);
            if (result) {
                logSpec = kOSKextLogErrorLevel;
            }
            OSKextLog(/* kext */ NULL, logSpec,
                "Child process %s[%d] exited with status %d.",
                argv0, child_pid, result);
        } else if (WIFSIGNALED(child_status)) {
            result = WTERMSIG(child_status);
            logSpec = kOSKextLogErrorLevel;
            OSKextLog(/* kext */ NULL, logSpec,
                "Child process %s[%d] exited due to signal %d.",
                argv0, child_pid, result);
        } else {
            // shouldn't be any other types of exit
            result = -1;
        }
    } else {
        result = child_pid;
    }

finish:
    setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, normal_iopolicy);

    return result;
}
