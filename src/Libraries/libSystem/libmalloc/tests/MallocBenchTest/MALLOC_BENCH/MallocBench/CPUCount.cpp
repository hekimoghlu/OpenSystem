/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#include "CPUCount.h"
#include <stdlib.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>

static size_t count;

size_t cpuCount()
{
    if (count)
        return count;

#ifdef __APPLE__
    size_t length = sizeof(count);
    int name[] = {
            CTL_HW,
            HW_NCPU
    };
    int sysctlResult = sysctl(name, sizeof(name) / sizeof(int), &count, &length, 0, 0);
    if (sysctlResult < 0)
        abort();
#else
    long sysconfResult = sysconf(_SC_NPROCESSORS_ONLN);
    if (sysconfResult < 0)
        abort();
    count = sysconfResult;
#endif

    return count;
}
