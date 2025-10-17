/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#pragma once

#include "BPlatform.h"

#if BPLATFORM(IOS_FAMILY)

#if __has_include(<System/sys/kern_memorystatus.h>)
extern "C" {
#include <System/sys/kern_memorystatus.h>
}
#else
extern "C" {

typedef struct memorystatus_memlimit_properties {
    int32_t memlimit_active;                /* jetsam memory limit (in MB) when process is active */
    uint32_t memlimit_active_attr;
    int32_t memlimit_inactive;              /* jetsam memory limit (in MB) when process is inactive */
    uint32_t memlimit_inactive_attr;
} memorystatus_memlimit_properties_t;

#define MEMORYSTATUS_CMD_GET_MEMLIMIT_PROPERTIES 8

}
#endif // __has_include(<System/sys/kern_memorystatus.h>)

extern "C" {
int memorystatus_control(uint32_t command, int32_t pid, uint32_t flags, void *buffer, size_t buffersize);
}

#endif // BPLATFORM(IOS_FAMILY)
