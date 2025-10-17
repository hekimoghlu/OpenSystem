/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#include "CacheModel.h"

#include <algorithm>
#include <wtf/RAMSize.h>
#include <wtf/Seconds.h>
#include <wtf/StdLibExtras.h>

namespace WebKit {

void calculateMemoryCacheSizes(CacheModel cacheModel, unsigned& cacheTotalCapacity, unsigned& cacheMinDeadCapacity, unsigned& cacheMaxDeadCapacity, Seconds& deadDecodedDataDeletionInterval, unsigned& backForwardCacheCapacity)
{
    uint64_t memorySize = ramSize() / MB;

    switch (cacheModel) {
    case CacheModel::DocumentViewer: {
        // back/forward cache capacity (in pages)
        backForwardCacheCapacity = 0;

        // Object cache capacities (in bytes)
        if (memorySize >= 2048)
            cacheTotalCapacity = 96 * MB;
        else if (memorySize >= 1536)
            cacheTotalCapacity = 64 * MB;
        else if (memorySize >= 1024)
            cacheTotalCapacity = 32 * MB;
        else if (memorySize >= 512)
            cacheTotalCapacity = 16 * MB;
        else
            cacheTotalCapacity = 8 * MB;

        cacheMinDeadCapacity = 0;
        cacheMaxDeadCapacity = 0;

        break;
    }
    case CacheModel::DocumentBrowser: {
        // back/forward cache capacity (in pages)
        if (memorySize >= 512)
            backForwardCacheCapacity = 2;
        else if (memorySize >= 256)
            backForwardCacheCapacity = 1;
        else
            backForwardCacheCapacity = 0;

        // Object cache capacities (in bytes)
        if (memorySize >= 2048)
            cacheTotalCapacity = 96 * MB;
        else if (memorySize >= 1536)
            cacheTotalCapacity = 64 * MB;
        else if (memorySize >= 1024)
            cacheTotalCapacity = 32 * MB;
        else if (memorySize >= 512)
            cacheTotalCapacity = 16 * MB;
        else
            cacheTotalCapacity = 8 * MB;

        cacheMinDeadCapacity = cacheTotalCapacity / 8;
        cacheMaxDeadCapacity = cacheTotalCapacity / 4;

        break;
    }
    case CacheModel::PrimaryWebBrowser: {
        // back/forward cache capacity (in pages)
        if (memorySize >= 512)
            backForwardCacheCapacity = 2;
        else if (memorySize >= 256)
            backForwardCacheCapacity = 1;
        else
            backForwardCacheCapacity = 0;

        // Object cache capacities (in bytes)
        // (Testing indicates that value / MB depends heavily on content and
        // browsing pattern. Even growth above 128MB can have substantial
        // value / MB for some content / browsing patterns.)
        if (memorySize >= 2048)
            cacheTotalCapacity = 128 * MB;
        else if (memorySize >= 1536)
            cacheTotalCapacity = 96 * MB;
        else if (memorySize >= 1024)
            cacheTotalCapacity = 64 * MB;
        else if (memorySize >= 512)
            cacheTotalCapacity = 32 * MB;
        else
            cacheTotalCapacity = 16 * MB;

        cacheMinDeadCapacity = cacheTotalCapacity / 4;
        cacheMaxDeadCapacity = cacheTotalCapacity / 2;

        // This code is here to avoid a PLT regression. We can remove it if we
        // can prove that the overall system gain would justify the regression.
        cacheMaxDeadCapacity = std::max(24u, cacheMaxDeadCapacity);

        deadDecodedDataDeletionInterval = 60_s;

        break;
    }
    default:
        ASSERT_NOT_REACHED();
    };
}

uint64_t calculateURLCacheDiskCapacity(CacheModel cacheModel, uint64_t diskFreeSize)
{
    uint64_t urlCacheDiskCapacity;

    switch (cacheModel) {
    case CacheModel::DocumentViewer: {
        // Disk cache capacity (in bytes)
        urlCacheDiskCapacity = 0;

        break;
    }
    case CacheModel::DocumentBrowser: {
        // Disk cache capacity (in bytes)
        if (diskFreeSize >= 16384)
            urlCacheDiskCapacity = 75 * MB;
        else if (diskFreeSize >= 8192)
            urlCacheDiskCapacity = 40 * MB;
        else if (diskFreeSize >= 4096)
            urlCacheDiskCapacity = 30 * MB;
        else
            urlCacheDiskCapacity = 20 * MB;

        break;
    }
    case CacheModel::PrimaryWebBrowser: {
        // Disk cache capacity (in bytes)
        if (diskFreeSize >= 16384)
            urlCacheDiskCapacity = 1 * GB;
        else if (diskFreeSize >= 8192)
            urlCacheDiskCapacity = 500 * MB;
        else if (diskFreeSize >= 4096)
            urlCacheDiskCapacity = 250 * MB;
        else if (diskFreeSize >= 2048)
            urlCacheDiskCapacity = 200 * MB;
        else if (diskFreeSize >= 1024)
            urlCacheDiskCapacity = 150 * MB;
        else
            urlCacheDiskCapacity = 100 * MB;

        break;
    }
    default:
        ASSERT_NOT_REACHED();
    };

    return urlCacheDiskCapacity;
}

} // namespace WebKit
