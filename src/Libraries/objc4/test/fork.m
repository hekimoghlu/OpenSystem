/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

// TEST_CONFIG OS=!exclavekit

#include "test.h"

void *flushthread(void *arg __unused)
{
    while (1) {
        _objc_flush_caches(nil);
    }
}

int main()
{
    pthread_t th;
    pthread_create(&th, nil, &flushthread, nil);

    alarm(120);
    
    [NSObject self];
    [NSObject self];

    int max = is_guardmalloc() ? 10: 100;
    
    for (int i = 0; i < max; i++) {
        pid_t child;
        switch ((child = fork())) {
          case -1:
            abort();
          case 0:
            // child
            alarm(10);
            [NSObject self];
            _exit(0);
          default: {
            // parent
            int result = 0;
            while (waitpid(child, &result, 0) < 0) {
                if (errno != EINTR) {
                    fail("waitpid failed (errno %d %s)", 
                         errno, strerror(errno));
                }
            }
            if (!WIFEXITED(result)) {
                fail("child crashed (waitpid result %d)", result);
            }

            [NSObject self];
            break;
          }
        }
    }

    succeed(__FILE__ " parent");
}
