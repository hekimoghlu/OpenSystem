/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#include <wtf/PageBlock.h>

#if OS(UNIX)
#include <unistd.h>
#endif

#if OS(WINDOWS)
#include <malloc.h>
#include <windows.h>
#endif

namespace WTF {

static size_t s_pageSize;

#if OS(UNIX)

inline size_t systemPageSize()
{
    return sysconf(_SC_PAGESIZE);
}

#elif OS(WINDOWS)

inline size_t systemPageSize()
{
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    return system_info.dwPageSize;
}

#endif

size_t pageSize()
{
    if (!s_pageSize) {
        s_pageSize = systemPageSize();
        RELEASE_ASSERT(isPowerOfTwo(s_pageSize));
        RELEASE_ASSERT_WITH_MESSAGE(s_pageSize <= CeilingOnPageSize, "CeilingOnPageSize is too low, raise it in PageBlock.h!");
        RELEASE_ASSERT(roundUpToMultipleOf(s_pageSize, CeilingOnPageSize) == CeilingOnPageSize);
    }
    return s_pageSize;
}

} // namespace WTF
