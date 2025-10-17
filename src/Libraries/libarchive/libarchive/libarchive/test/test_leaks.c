/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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

#include "test.h"
__FBSDID("$FreeBSD$");

#include <spawn.h>

extern char **environ;

DEFINE_TEST(test_leaks)
{
    char execString[16] = {'\0'};
    snprintf(execString, sizeof(execString), "%d", getpid());
    char *args[] = { "/usr/bin/leaks", execString, NULL };
    const char *memgraphError = "Leaks found, but error occurred while generating memgraph";
    
    pid_t pid = -1;
    int status = 0;
    int rv = 0;
    int exitCode = -1;
    
    rv = posix_spawn(&pid, args[0], NULL, NULL, args, environ);
    assert(!rv);
    
    do {
        rv = waitpid(pid, &status, 0);
    } while (rv < 0 && errno == EINTR);
    
    if(WIFEXITED(status)) {
        exitCode = WEXITSTATUS(status);
    }
    
    if (!exitCode) {
        // No leaks found
        return;
    }
    
    // Leaks found. Generate memgraph.
    char *memgraphArgs[] = { "/usr/bin/leaks", execString, "-outputGraph=leaks-libarchive.memgraph", NULL };
    rv = posix_spawn(&pid, memgraphArgs[0], NULL, NULL, memgraphArgs, environ);
    failure("%s", memgraphError);
    assert(!rv);
    
    do {
        rv = waitpid(pid, &status, 0);
    } while (rv < 0 && errno == EINTR);
    
    if(WIFEXITED(status)) {
        failure("%s", memgraphError);
        assert(WEXITSTATUS(status) < 1);
    }
    
    failure("Leaks found");
    assert(!exitCode);
}
