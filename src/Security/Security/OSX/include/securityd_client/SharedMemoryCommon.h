/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#ifndef __SHARED_MEMORY_COMMON__
#define __SHARED_MEMORY_COMMON__



#include <sys/types.h>

const unsigned kSegmentSize = 4096;
const unsigned kNumberOfSegments = 8;
const unsigned kSharedMemoryPoolSize = kSegmentSize * kNumberOfSegments;

const unsigned kBytesWrittenOffset = 0;
const unsigned kBytesWrittenLength = 4;
const unsigned kPoolAvailableForData = kSharedMemoryPoolSize - kBytesWrittenLength;

typedef u_int32_t SegmentOffsetType;

class SharedMemoryCommon
{
public:
    SharedMemoryCommon() {}
    virtual ~SharedMemoryCommon ();

    // Is this a system user or a regular user?
    static uid_t fixUID(uid_t uid) { return (uid < 500) ? 0 : uid; }

    static std::string SharedMemoryFilePath(const char *segmentName, uid_t uid);
    static std::string notificationDescription(int domain, int event);

    constexpr static const char* const kMDSDirectory = "/private/var/db/mds/";
    constexpr static const char* const kMDSMessagesDirectory = "/private/var/db/mds/messages/";
    constexpr static const char* const kUserPrefix = "se_";

    constexpr static const char* const kDefaultSecurityMessagesName = "SecurityMessages";
};


#endif
