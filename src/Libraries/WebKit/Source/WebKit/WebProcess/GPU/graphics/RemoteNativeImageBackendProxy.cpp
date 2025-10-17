/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
#include "RemoteNativeImageBackendProxy.h"

#include "RemoteResourceCacheProxy.h"

namespace WebKit {
using namespace WebCore;

std::unique_ptr<RemoteNativeImageBackendProxy> RemoteNativeImageBackendProxy::create(NativeImage& image, const DestinationColorSpace& colorSpace)
{
    RefPtr<ShareableBitmap> bitmap;
    PlatformImagePtr platformImage;
#if USE(CG)
    bitmap = ShareableBitmap::createFromImagePixels(image);
    if (bitmap)
        platformImage = bitmap->createPlatformImage(DontCopyBackingStore, ShouldInterpolate::Yes);
#endif

    // If we failed to create ShareableBitmap or PlatformImage, fall back to image-draw method.
    if (!platformImage) {
        bitmap = ShareableBitmap::createFromImageDraw(image, colorSpace);
        if (bitmap)
            platformImage = bitmap->createPlatformImage(DontCopyBackingStore, ShouldInterpolate::Yes);

        // If createGraphicsContext() failed because the image colorSpace is not
        // supported for output, fallback to SRGB.
        if (!platformImage) {
            bitmap = ShareableBitmap::createFromImageDraw(image, DestinationColorSpace::SRGB());
            if (bitmap)
                platformImage = bitmap->createPlatformImage(DontCopyBackingStore, ShouldInterpolate::Yes);
        }
    }
    if (!platformImage)
        return nullptr;
    return std::unique_ptr<RemoteNativeImageBackendProxy> { new RemoteNativeImageBackendProxy(bitmap.releaseNonNull(), WTFMove(platformImage)) };
}

RemoteNativeImageBackendProxy::RemoteNativeImageBackendProxy(Ref<ShareableBitmap> bitmap, PlatformImagePtr platformImage)
    : m_bitmap(WTFMove(bitmap))
    , m_platformBackend(WTFMove(platformImage))
{
}

RemoteNativeImageBackendProxy::~RemoteNativeImageBackendProxy() = default;

const PlatformImagePtr& RemoteNativeImageBackendProxy::platformImage() const
{
    return m_platformBackend.platformImage();
}

IntSize RemoteNativeImageBackendProxy::size() const
{
    return m_platformBackend.size();
}

bool RemoteNativeImageBackendProxy::hasAlpha() const
{
    return m_platformBackend.hasAlpha();
}

DestinationColorSpace RemoteNativeImageBackendProxy::colorSpace() const
{
    return m_platformBackend.colorSpace();
}

Headroom RemoteNativeImageBackendProxy::headroom() const
{
    return m_platformBackend.headroom();
}

bool RemoteNativeImageBackendProxy::isRemoteNativeImageBackendProxy() const
{
    return true;
}

std::optional<ShareableBitmap::Handle> RemoteNativeImageBackendProxy::createHandle()
{
    return m_bitmap->createHandle();
}

}

#endif
