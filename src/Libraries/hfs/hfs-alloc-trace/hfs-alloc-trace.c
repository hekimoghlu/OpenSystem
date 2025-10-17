/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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

//
//  hfs-alloc-trace.c
//  hfs-alloc-trace
//
//  Created by Chris Suter on 8/19/15.
//
//

#include <sys/sysctl.h>
#include <stdlib.h>
#include <err.h>
#include <stdio.h>
#include <stdbool.h>

#include "../core/hfs_alloc_trace.h"

int main(void)
{
    size_t sz = 128 * 1024;
    struct hfs_alloc_trace_info *info = malloc(sz);

    if (sysctlbyname("vfs.generic.hfs.alloc_trace_info", info, &sz,
                     NULL, 0)) {
        err(1, "sysctlbyname failed");
    }

    for (int i = 0; i < info->entry_count; ++i) {
        printf(" -- 0x%llx:%llu <%llu> --\n", info->entries[i].ptr,
               info->entries[i].sequence, info->entries[i].size);
        for (int j = 0; j < HFS_ALLOC_BACKTRACE_LEN; ++j)
            printf("0x%llx\n", info->entries[i].backtrace[j]);
    }

    if (info->more)
        printf("[skipped]\n");

    return 0;
}
