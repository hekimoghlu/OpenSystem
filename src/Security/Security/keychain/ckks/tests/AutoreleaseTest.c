/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <dispatch/dispatch.h>
#include <objc/objc-internal.h>

#include "AutoreleaseTest.h"

static void
read_releases_pending(int fd, void (^handler)(ssize_t))
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        ssize_t result = -1;

        FILE *fp = fdopen(fd, "r");

        char *line = NULL;
        size_t linecap = 0;
        while (getline(&line, &linecap, fp) > 0) {
            ssize_t pending;

            if (sscanf(line, "objc[%*d]: %ld releases pending", &pending) == 1) {
                result = pending;
                break;
            }
        }
        free(line);

        fclose(fp);

        handler(result);
    });
}

ssize_t
pending_autorelease_count(void)
{
    __block ssize_t result = -1;
    dispatch_semaphore_t sema;
    int fds[2];
    int saved_stderr;

    // stderr replacement pipe
    pipe(fds);
    fcntl(fds[1], F_SETNOSIGPIPE, 1);

    // sead asynchronously - takes ownership of fds[0]
    sema = dispatch_semaphore_create(0);
    read_releases_pending(fds[0], ^(ssize_t pending) {
        result = pending;
        dispatch_semaphore_signal(sema);
    });

    // save and replace stderr
    saved_stderr = dup(STDERR_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);

    // make objc print the current autorelease pool
    _objc_autoreleasePoolPrint();

    // restore stderr
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);

    // wait for the reader
    dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
#if !__has_feature(objc_arc)
    dispatch_release(sema);
#endif

    return result;
}
