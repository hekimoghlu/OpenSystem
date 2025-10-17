/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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

#if ENABLE(GPU_PROCESS)
#include "RemoteSharedResourceCache.h"

#include "GPUConnectionToWebProcess.h"
#include <wtf/TZoneMallocInlines.h>

#if HAVE(IOSURFACE)
#include <WebCore/IOSurfacePool.h>
#endif

namespace WebKit {
using namespace WebCore;

constexpr Seconds defaultRemoteSharedResourceCacheTimeout = 15_s;

// Per GPU process limit of accelerated image buffers. These consume limited global OS resources.
constexpr size_t globalAcceleratedImageBufferLimit = 10000;

// Per GPU process limit of image buffers for canvas. These consume limited process-wide resources.
constexpr size_t globalImageBufferForCanvasLimit = 200000;

// Per Web Content process limit of accelerated image buffers for canvas. Prevents GPU resource exhaustion affecting other Web Content processes.
constexpr size_t acceleratedImageBufferForCanvasLimit = 5000;

// Per Web Content process limit of image buffers for canvas. Prevents IPC-related resource exhaustion affecting other Web Content processes.
constexpr size_t imageBufferForCanvasLimit = 50000;

static std::atomic<size_t> globalAcceleratedImageBufferCount = 0;
static std::atomic<size_t> globalImageBufferForCanvasCount = 0;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteSharedResourceCache);

Ref<RemoteSharedResourceCache> RemoteSharedResourceCache::create(GPUConnectionToWebProcess& gpuConnectionToWebProcess)
{
    return adoptRef(*new RemoteSharedResourceCache(gpuConnectionToWebProcess));
}

RemoteSharedResourceCache::RemoteSharedResourceCache(GPUConnectionToWebProcess& gpuConnectionToWebProcess)
    : m_resourceOwner(gpuConnectionToWebProcess.webProcessIdentity())
#if HAVE(IOSURFACE)
    , m_ioSurfacePool(IOSurfacePool::create())
#endif
{
}

RemoteSharedResourceCache::~RemoteSharedResourceCache() = default;

void RemoteSharedResourceCache::addSerializedImageBuffer(RenderingResourceIdentifier identifier, Ref<ImageBuffer> imageBuffer)
{
    m_serializedImageBuffers.add({ identifier, 0 }, WTFMove(imageBuffer));
}

RefPtr<ImageBuffer> RemoteSharedResourceCache::takeSerializedImageBuffer(RenderingResourceIdentifier identifier)
{
    return m_serializedImageBuffers.take({ { identifier, 0 }, 0 }, defaultRemoteSharedResourceCacheTimeout);
}

void RemoteSharedResourceCache::releaseSerializedImageBuffer(WebCore::RenderingResourceIdentifier identifier)
{
    m_serializedImageBuffers.remove({ { identifier, 0 }, 0 });
}

void RemoteSharedResourceCache::lowMemoryHandler()
{
#if HAVE(IOSURFACE)
    Ref { m_ioSurfacePool }->discardAllSurfaces();
#endif
}

void RemoteSharedResourceCache::didCreateImageBuffer(RenderingPurpose purpose, RenderingMode renderingMode)
{
    if (purpose == RenderingPurpose::Canvas) {
        if (renderingMode == RenderingMode::Accelerated)
            ++m_acceleratedImageBufferForCanvasCount;
        ++globalImageBufferForCanvasCount;
        ++m_imageBufferForCanvasCount;
    }
    if (renderingMode == RenderingMode::Accelerated)
        ++globalAcceleratedImageBufferCount;
}

void RemoteSharedResourceCache::didReleaseImageBuffer(RenderingPurpose purpose, RenderingMode renderingMode)
{
    if (purpose == RenderingPurpose::Canvas) {
        if (renderingMode == RenderingMode::Accelerated)
            --m_acceleratedImageBufferForCanvasCount;
        --globalImageBufferForCanvasCount;
        --m_imageBufferForCanvasCount;
    }
    if (renderingMode == RenderingMode::Accelerated)
        --globalAcceleratedImageBufferCount;
}

bool RemoteSharedResourceCache::reachedAcceleratedImageBufferLimit(RenderingPurpose purpose) const
{
    return (purpose == RenderingPurpose::Canvas && m_acceleratedImageBufferForCanvasCount >= acceleratedImageBufferForCanvasLimit) || globalAcceleratedImageBufferCount >= globalAcceleratedImageBufferLimit;
}

bool RemoteSharedResourceCache::reachedImageBufferForCanvasLimit() const
{
    return m_imageBufferForCanvasCount >= imageBufferForCanvasLimit || globalImageBufferForCanvasCount >= globalImageBufferForCanvasLimit;
}

ImageBufferResourceLimits RemoteSharedResourceCache::getResourceLimitsForTesting() const
{
    return {
        .acceleratedImageBufferForCanvasCount = m_acceleratedImageBufferForCanvasCount,
        .acceleratedImageBufferForCanvasLimit = acceleratedImageBufferForCanvasLimit,
        .globalAcceleratedImageBufferCount = globalAcceleratedImageBufferCount,
        .globalAcceleratedImageBufferLimit = globalAcceleratedImageBufferLimit,
        .globalImageBufferForCanvasCount = globalImageBufferForCanvasCount,
        .globalImageBufferForCanvasLimit = globalImageBufferForCanvasLimit,
        .imageBufferForCanvasCount = m_imageBufferForCanvasCount,
        .imageBufferForCanvasLimit = imageBufferForCanvasLimit,
    };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
