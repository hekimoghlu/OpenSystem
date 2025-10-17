/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#include "ImageBufferRemoteIOSurfaceBackend.h"

#if HAVE(IOSURFACE)

#include "Logging.h"
#include <WebCore/GraphicsContextCG.h>
#include <WebCore/ImageBufferIOSurfaceBackend.h>
#include <WebCore/PixelBuffer.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBufferRemoteIOSurfaceBackend);

IntSize ImageBufferRemoteIOSurfaceBackend::calculateSafeBackendSize(const Parameters& parameters)
{
    return ImageBufferIOSurfaceBackend::calculateSafeBackendSize(parameters);
}

size_t ImageBufferRemoteIOSurfaceBackend::calculateMemoryCost(const Parameters& parameters)
{
    return ImageBufferIOSurfaceBackend::calculateMemoryCost(parameters);
}

std::unique_ptr<ImageBufferRemoteIOSurfaceBackend> ImageBufferRemoteIOSurfaceBackend::create(const Parameters& parameters, ImageBufferBackendHandle handle)
{
    if (!std::holds_alternative<MachSendRight>(handle)) {
        RELEASE_ASSERT_NOT_REACHED();
        return nullptr;
    }

    return makeUnique<ImageBufferRemoteIOSurfaceBackend>(parameters, WTFMove(std::get<MachSendRight>(handle)));
}

std::optional<ImageBufferBackendHandle> ImageBufferRemoteIOSurfaceBackend::createBackendHandle(SharedMemory::Protection) const
{
    return MachSendRight { m_handle };
}

std::optional<ImageBufferBackendHandle> ImageBufferRemoteIOSurfaceBackend::takeBackendHandle(SharedMemory::Protection)
{
    return std::exchange(m_handle, { });
}

void ImageBufferRemoteIOSurfaceBackend::setBackendHandle(ImageBufferBackendHandle&& handle)
{
    if (!std::holds_alternative<MachSendRight>(handle)) {
        RELEASE_ASSERT_NOT_REACHED();
        return;
    }
    m_handle = WTFMove(std::get<MachSendRight>(handle));
}

void ImageBufferRemoteIOSurfaceBackend::clearBackendHandle()
{
    m_handle = { };
}

bool ImageBufferRemoteIOSurfaceBackend::canMapBackingStore() const
{
    return false;
}

GraphicsContext& ImageBufferRemoteIOSurfaceBackend::context()
{
    RELEASE_ASSERT_NOT_REACHED();
    return *(GraphicsContext*)nullptr;
}

unsigned ImageBufferRemoteIOSurfaceBackend::bytesPerRow() const
{
    return ImageBufferIOSurfaceBackend::calculateBytesPerRow(m_parameters.backendSize);
}

RefPtr<NativeImage> ImageBufferRemoteIOSurfaceBackend::copyNativeImage()
{
    RELEASE_ASSERT_NOT_REACHED();
    return { };
}

RefPtr<NativeImage> ImageBufferRemoteIOSurfaceBackend::createNativeImageReference()
{
    RELEASE_ASSERT_NOT_REACHED();
    return { };
}

void ImageBufferRemoteIOSurfaceBackend::getPixelBuffer(const IntRect&, PixelBuffer&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void ImageBufferRemoteIOSurfaceBackend::putPixelBuffer(const PixelBuffer&, const IntRect&, const IntPoint&, AlphaPremultiplication)
{
    RELEASE_ASSERT_NOT_REACHED();
}

String ImageBufferRemoteIOSurfaceBackend::debugDescription() const
{
    TextStream stream;
    stream << "ImageBufferRemoteIOSurfaceBackend " << this << " handle " << m_handle.sendRight() << " " << m_volatilityState;
    return stream.release();
}


} // namespace WebKit

#endif // HAVE(IOSURFACE)
