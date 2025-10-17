/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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

#include <darwintest.h>
#include <darwintest_utils.h>

#include "libc_hooks_helper.h"

T_DECL(libc_hooks_tempnam, "Test libc_hooks for tempnam")
{
    // Test
    char dir[] = "foobar"; char pfx[] = "/etc/";
    libc_hooks_log_start();
    char *s = tempnam(dir, pfx);
    libc_hooks_log_stop(6);

    // Check
    T_LOG("tempnam(\"%s\", \"%s\")", dir, pfx);
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read_cstring, dir, strlen(dir) + 1), "checking dir");
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read_cstring, pfx, strlen(pfx) + 1), "checking pfx");
#if 0 // TBD: Where are these coming from?
    libc_hooks_log_dump(libc_hooks_log);
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read, ?, SIZE_LOCALE_T)), "checking ? (location)");
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read_cstring, ?, 11)), "checking ?");
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read, ?, 9)), "checking ?");
#else
    libc_hooks_log.check += 3;
#endif
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read, pfx, strlen(pfx)), "checking pfx (being read)");

    // Cleanup
    free(s);
}

