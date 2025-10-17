/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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

T_DECL(libc_hooks_freopen, "Test libc_hooks for freopen")
{
    char file[] = "/dev/null"; char mode1[] = "w"; char mode2[] = "r";

    // Setup
    T_SETUPBEGIN;
    FILE *f1 = fopen(file, mode1);
    T_SETUPEND;

    // Test
    libc_hooks_log_start();
    FILE *f2 = freopen(file, mode2, f1);
    libc_hooks_log_stop(3);

    // Check
    T_LOG("freopen(\"%s\", \"%s\", f2)", file, mode2, f1);

    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read_cstring, file, strlen(file) + 1), "checking file");
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_read_cstring, mode2, strlen(mode2) + 1), "checking mode");
    libc_hooks_log_expect(LIBC_HOOKS_LOG(libc_hooks_will_write, f2, sizeof(*f2)), "checking f");

    // Cleanup
    fclose(f1);
    fclose(f2);
}
