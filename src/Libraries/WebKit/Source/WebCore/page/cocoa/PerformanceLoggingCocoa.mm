/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#import "config.h"
#import "PerformanceLogging.h"

#import <mach/mach.h>
#import <mach/task_info.h>

namespace WebCore {

std::optional<uint64_t> PerformanceLogging::physicalFootprint()
{
    task_vm_info_data_t vmInfo;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t result = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vmInfo, &count);
    if (result != KERN_SUCCESS)
        return std::nullopt;
    return vmInfo.phys_footprint;
}

void PerformanceLogging::getPlatformMemoryUsageStatistics(Vector<std::pair<ASCIILiteral, size_t>>& stats)
{
    task_vm_info_data_t vmInfo;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t err = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vmInfo, &count);
    if (err != KERN_SUCCESS)
        return;
    stats.append(std::pair { "internal_mb"_s, static_cast<size_t>(vmInfo.internal >> 20) });
    stats.append(std::pair { "compressed_mb"_s, static_cast<size_t>(vmInfo.compressed >> 20) });
    stats.append(std::pair { "phys_footprint_mb"_s, static_cast<size_t>(vmInfo.phys_footprint >> 20) });
    stats.append(std::pair { "resident_size_mb"_s, static_cast<size_t>(vmInfo.resident_size >> 20) });
    stats.append(std::pair { "virtual_size_mb"_s, static_cast<size_t>(vmInfo.virtual_size >> 20) });
}

}
