/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#include "Memory.h"
#include <iostream>
#include <stdlib.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#else
#include <stdio.h>
#include <unistd.h>
#endif

Memory currentMemoryBytes()
{
    Memory memory;

#ifdef __APPLE__
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t vm_size = TASK_VM_INFO_COUNT;
    if (KERN_SUCCESS != task_info(mach_task_self(), TASK_VM_INFO_PURGEABLE, (task_info_t)(&vm_info), &vm_size)) {
        std::cout << "Failed to get mach task info" << std::endl;
        exit(1);
    }

    memory.resident = vm_info.internal + vm_info.compressed - vm_info.purgeable_volatile_pmap;
    memory.residentMax = vm_info.resident_size_peak;
#else
    FILE* file = fopen("/proc/self/status", "r");

    auto forEachLine = [] (FILE* file, auto functor) {
        char* buffer = nullptr;
        size_t size = 0;
        while (getline(&buffer, &size, file) != -1) {
            functor(buffer, size);
            ::free(buffer); // Be careful. getline's memory allocation is done by system malloc.
            buffer = nullptr;
            size = 0;
        }
    };

    unsigned long vmHWM = 0;
    unsigned long vmRSS = 0;
    unsigned long rssFile = 0;
    unsigned long rssShmem = 0;
    forEachLine(file, [&] (char* buffer, size_t) {
        unsigned long sizeInKB = 0;
        if (sscanf(buffer, "VmHWM: %lu kB", &sizeInKB) == 1)
            vmHWM = sizeInKB * 1024;
        else if (sscanf(buffer, "VmRSS: %lu kB", &sizeInKB) == 1)
            vmRSS = sizeInKB * 1024;
        else if (sscanf(buffer, "RssFile: %lu kB", &sizeInKB) == 1)
            rssFile = sizeInKB * 1024;
        else if (sscanf(buffer, "RssShmem: %lu kB", &sizeInKB) == 1)
            rssShmem = sizeInKB * 1024;
    });
    fclose(file);
    memory.resident = vmRSS - (rssFile + rssShmem);
    memory.residentMax = vmHWM - (rssFile + rssShmem); // We do not have any way to get the peak of RSS of anonymous pages. Here, we subtract RSS of files and shmem to estimate the peak of RSS of anonymous pages.
#endif
    return memory;
}
