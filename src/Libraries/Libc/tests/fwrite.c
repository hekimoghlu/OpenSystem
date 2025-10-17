/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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

T_DECL(libc_hooks_fwrite, "Test libc_hooks for fwrite")
{
    // Setup
    T_SETUPBEGIN;
    FILE *f = fopen("/dev/null", "w");
    T_SETUPEND;

    // Test
    char buf[256] = "foo";
    libc_hooks_log_start();
    fwrite(buf, sizeof(buf), 1, f);
    libc_hooks_log_stop(2);

    // Check
    T_LOG("fwrite(\"%s\", %zu, %zu, f)", buf, sizeof(buf), 1UL);
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_write, f, sizeof(*f)), "checking f");
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read, buf, sizeof(buf)), "checking buf");

    // Cleanup
    fclose(f);
}
