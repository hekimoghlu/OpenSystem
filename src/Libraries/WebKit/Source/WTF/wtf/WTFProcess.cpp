/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
#include "config.h"
#include <wtf/WTFProcess.h>

#include <stdlib.h>

#if OS(UNIX)
#include <unistd.h>
#endif

#if OS(WINDOWS)
#include <windows.h>
#endif

namespace WTF {

void exitProcess(int status)
{
#if OS(WINDOWS)
    // Windows do not have "main thread" concept. So, shutdown of the main thread does not mean immediate process shutdown.
    // As a result, if there is running thread, it sometimes cause deadlock.
    //
    // > If one of the terminated threads in the process holds a lock and the DLL detach code in one of the loaded DLLs
    // > attempts to acquire the same lock, then calling ExitProcess results in a deadlock. In contrast, if a process
    // > terminates by calling TerminateProcess, the DLLs that the process is attached to are not notified of the process
    // > termination. Therefore, if you do not know the state of all threads in your process, it is better to call TerminateProcess
    // > than ExitProcess. Note that returning from the main function of an application results in a call to ExitProcess.
    // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-exitprocess
    //
    // Always use TerminateProcess since framework does not know the status of the other threads.
    TerminateProcess(GetCurrentProcess(), status);

    // The code can reach here only when very buggy anti-virus software hooks are integrated into the system. To make it safe, in that case,
    // let the process crash explicitly.
    CRASH();
#else
    exit(status);
#endif
}

void terminateProcess(int status)
{
#if OS(WINDOWS)
    // On Windows, exitProcess and terminateProcess do the same thing due to its more complicated main thread handling.
    // See comment in exitProcess.
    exitProcess(status);
#else
    _exit(status);
#endif
}

}
