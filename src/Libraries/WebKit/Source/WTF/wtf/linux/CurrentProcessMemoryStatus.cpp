/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#include <wtf/linux/CurrentProcessMemoryStatus.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <wtf/PageBlock.h>

namespace WTF {

void currentProcessMemoryStatus(ProcessMemoryStatus& memoryStatus)
{
    FILE* file = fopen("/proc/self/statm", "r");
    if (!file)
        return;

    char buffer[128];
    char* line = fgets(buffer, 128, file);
    fclose(file);
    if (!line)
        return;

    size_t pageSize = WTF::pageSize();
    char* end = nullptr;
    unsigned long long intValue = strtoull(line, &end, 10);
    memoryStatus.size = intValue * pageSize;
    intValue = strtoull(end, &end, 10);
    memoryStatus.resident = intValue * pageSize;
    intValue = strtoull(end, &end, 10);
    memoryStatus.shared = intValue * pageSize;
    intValue = strtoull(end, &end, 10);
    memoryStatus.text = intValue * pageSize;
    intValue = strtoull(end, &end, 10);
    memoryStatus.lib = intValue * pageSize;
    intValue = strtoull(end, &end, 10);
    memoryStatus.data = intValue * pageSize;
    intValue = strtoull(end, &end, 10);
    memoryStatus.dt = intValue * pageSize;
}

} // namespace WTF
