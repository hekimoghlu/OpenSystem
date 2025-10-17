/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#import "DiskCacheMonitorCocoa.h"

#import "CachedResource.h"
#import "MemoryCache.h"
#import "SharedBuffer.h"
#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/MainThread.h>
#import <wtf/RefPtr.h>

#if USE(WEB_THREAD)
#import "WebCoreThreadRun.h"
#endif

namespace WebCore {

// The maximum number of seconds we'll try to wait for a resource to be disk cached before we forget the request.
static const double diskCacheMonitorTimeout = 20;

RefPtr<SharedBuffer> DiskCacheMonitor::tryGetFileBackedSharedBufferFromCFURLCachedResponse(CFCachedURLResponseRef cachedResponse)
{
    auto data = _CFCachedURLResponseGetMemMappedData(cachedResponse);
    if (!data)
        return nullptr;

    return SharedBuffer::create(data);
}

void DiskCacheMonitor::monitorFileBackingStoreCreation(const ResourceRequest& request, PAL::SessionID sessionID, CFCachedURLResponseRef cachedResponse)
{
    if (!cachedResponse)
        return;

    // FIXME: It's not good to have the new here, but the delete inside the constructor. Reconsider this design.
    new DiskCacheMonitor(request, sessionID, cachedResponse); // Balanced by delete and unique_ptr in the blocks set up in the constructor, one of which is guaranteed to run.
}

DiskCacheMonitor::DiskCacheMonitor(const ResourceRequest& request, PAL::SessionID sessionID, CFCachedURLResponseRef cachedResponse)
    : m_resourceRequest(request)
    , m_sessionID(sessionID)
{
    ASSERT(isMainThread());

    // Set up a delayed callback to cancel this monitor if the resource hasn't been cached yet.
    __block DiskCacheMonitor* rawMonitor = this;
    auto cancelMonitorBlock = ^{
        delete rawMonitor; // Balanced by "new DiskCacheMonitor" in monitorFileBackingStoreCreation.
        rawMonitor = nullptr;
    };

#if USE(WEB_THREAD)
    auto cancelMonitorBlockToRun = ^{
        WebThreadRun(cancelMonitorBlock);
    };
#else
    auto cancelMonitorBlockToRun = cancelMonitorBlock;
#endif

    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC * diskCacheMonitorTimeout), dispatch_get_main_queue(), cancelMonitorBlockToRun);

    // Set up the disk caching callback to create the ShareableResource and send it to the WebProcess.
    auto block = ^(CFCachedURLResponseRef cachedResponse) {
        ASSERT(isMainThread());
        // If the monitor isn't there then it timed out before this resource was cached to disk.
        if (!rawMonitor)
            return;

        auto monitor = std::unique_ptr<DiskCacheMonitor>(rawMonitor); // Balanced by "new DiskCacheMonitor" in monitorFileBackingStoreCreation.
        rawMonitor = nullptr;

        auto fileBackedBuffer = DiskCacheMonitor::tryGetFileBackedSharedBufferFromCFURLCachedResponse(cachedResponse);
        if (!fileBackedBuffer)
            return;

        monitor->resourceBecameFileBacked(*fileBackedBuffer);
    };

#if USE(WEB_THREAD)
    auto blockToRun = ^(CFCachedURLResponseRef response) {
        auto strongResponse = retainPtr(response);
        WebThreadRun(^{
            block(strongResponse.get());
        });
    };
#else
    auto blockToRun = block;
#endif
    _CFCachedURLResponseSetBecameFileBackedCallBackBlock(cachedResponse, blockToRun, dispatch_get_main_queue());
}

void DiskCacheMonitor::resourceBecameFileBacked(SharedBuffer& fileBackedBuffer)
{
    CachedResourceHandle resource = MemoryCache::singleton().resourceForRequest(m_resourceRequest, m_sessionID);
    if (!resource)
        return;

    resource->tryReplaceEncodedData(fileBackedBuffer);
}


} // namespace WebCore
