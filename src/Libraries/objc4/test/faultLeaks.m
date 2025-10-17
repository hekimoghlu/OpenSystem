/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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

// TEST_CONFIG MEM=mrc, LANGUAGE=objective-c
// TEST_ENV OBJC_DEBUG_SYNC_ERRORS=Fault
/* TEST_RUN_OUTPUT
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
objc\[\d+\]: objc_sync_exit\(0x[a-fA-F0-9]+\) returned error -1
[\S\s]*0 leaks for 0 total leaked bytes[\S\s]*
OK: faultLeaks.m
END
*/

#include <objc/objc-sync.h>

#include <spawn.h>
#include <stdio.h>

#include "test.h"
#include "testroot.i"

int main() {
    id obj = [TestRoot alloc];

    // objc_sync_exit on an object that isn't locked will provoke a fault from
    // OBJC_DEBUG_SYNC_ERRORS=Fault. Do this several times to ensure any leak is
    // detected.
    objc_sync_exit(obj);
    objc_sync_exit(obj);
    objc_sync_exit(obj);
    objc_sync_exit(obj);
    objc_sync_exit(obj);
    objc_sync_exit(obj);
    objc_sync_exit(obj);
    objc_sync_exit(obj);
    objc_sync_exit(obj);
    objc_sync_exit(obj);

    char *pidstr;
    int result = asprintf(&pidstr, "%u", getpid());
    testassert(result);

    extern char **environ;
    char *argv[] = { "/usr/bin/leaks", pidstr, NULL };
    pid_t pid;
    result = posix_spawn(&pid, "/usr/bin/leaks", NULL, NULL, argv, environ);
    if (result) {
        perror("posix_spawn");
        exit(1);
    }
    wait4(pid, NULL, 0, NULL);

    free(pidstr);
    [obj release];

    succeed(__FILE__);
}
