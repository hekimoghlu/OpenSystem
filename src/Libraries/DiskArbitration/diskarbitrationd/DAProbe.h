/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#ifndef __DISKARBITRATIOND_DAPROBE__
#define __DISKARBITRATIOND_DAPROBE__

#include <CoreFoundation/CoreFoundation.h>

#include "DADisk.h"
#include "DAFileSystem.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void ( *DAProbeCallback )( int             status,
                                   DAFileSystemRef filesystem,
                                   int             cleanStatus,
                                   CFStringRef     name,
                                   CFStringRef     type,
                                   CFUUIDRef       uuid,
                                   void *          context );

typedef struct __DAProbeCallbackContext __DAProbeCallbackContext;

struct __DAProbeCallbackContext
{
    DAProbeCallback   callback;
    void *            callbackContext;
    CFMutableArrayRef candidates;
    DADiskRef         disk;
    DADiskRef         containerDisk;
    DAFileSystemRef   filesystem;
    uint64_t          startTime;
#ifdef DA_FSKIT
    int               gotFSModules;
#endif
};

extern void DAProbe( DADiskRef       disk,
                     DADiskRef containerDisk,
                     DAProbeCallback callback,
                     void *          callbackContext );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DAPROBE__ */
