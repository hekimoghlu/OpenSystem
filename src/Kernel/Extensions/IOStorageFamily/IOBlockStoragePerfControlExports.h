/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
#ifndef IOBlockStoragePerfControlExports_h
#define IOBlockStoragePerfControlExports_h

#include <IOKit/perfcontrol/IOPerfControl.h>

#define IOBlockStorageWorkArgsVersion2        2
#define IOBlockStorageCurrentWorkArgsVersion  IOBlockStorageWorkArgsVersion2

struct IOBlockStorageWorkFlags
{
    /* isRead is True for read op */
    bool isRead{};

    /* isLowPriority is True for low priority IOs */
    bool isLowPriority{};

    /* Size of the I/O in bytes */
    uint64_t ioSize{};
};

struct IOBlockStorageWorkEndArgs
{
    /* Timestamp taken after the completion handler has finished */
    uint64_t postCompletionTime;
};

#endif /* IOBlockStoragePerfControlExports_h */
