/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include "MachUtilities.h"

#include <mach/mach_init.h>
#include <mach/task.h>

void setMachPortQueueLength(mach_port_t receivePort, mach_port_msgcount_t queueLength)
{
    ASSERT(MACH_PORT_VALID(receivePort));

    mach_port_limits_t portLimits;
    portLimits.mpl_qlimit = queueLength;

    mach_port_set_attributes(mach_task_self(), receivePort, MACH_PORT_LIMITS_INFO, reinterpret_cast<mach_port_info_t>(&portLimits), MACH_PORT_LIMITS_INFO_COUNT);
}
