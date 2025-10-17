/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
#include "WebImage.h"

#include "ImageBufferShareableBitmapBackend.h"
#include <WebCore/ChromeClient.h>
#include <WebCore/ImageBuffer.h>

namespace WebKit {
using namespace WebCore;

Ref<WebImage> WebImage::createEmpty()
{
    return adoptRef(*new WebImage(nullptr));
}

Ref<WebImage> WebImage::create(const IntSize& size, ImageOptions options, const DestinationColorSpace& colorSpace, ChromeClient* client)
{
    if (client) {
        auto purpose = options.contains(ImageOption::Shareable) ? RenderingPurpose::ShareableSnapshot : RenderingPurpose::Snapshot;
        purpose = options.contains(ImageOption::Local) ? RenderingPurpose::ShareableLocalSnapshot : purpose;
        
        if (auto buffer = client->createImageBuffer(size, RenderingMode::Unaccelerated, purpose, 1, colorSpace, ImageBufferPixelFormat::BGRA8))
            return WebImage::create(buffer.releaseNonNull());
    }

    if (options.contains(ImageOption::Shareable)) {
        auto buffer = ImageBuffer::create<ImageBufferShareableBitmapBackend>(size, 1, colorSpace, ImageBufferPixelFormat::BGRA8, RenderingPurpose::ShareableSnapshot, { });
        if (!buffer)
            return createEmpty();
        return WebImage::create(buffer.releaseNonNull());
    }

    auto buffer = ImageBuffer::create(size, RenderingMode::Unaccelerated, RenderingPurpose::Snapshot, 1, colorSpace, ImageBufferPixelFormat::BGRA8);
    if (!buffer)
        return createEmpty();
    return WebImage::create(buffer.releaseNonNull());
}

Ref<WebImage> WebImage::create(std::optional<ParametersAndHandle>&& parametersAndHandle)
{
    if (!parametersAndHandle)
        return createEmpty();
    auto [parameters, handle] = WTFMove(*parametersAndHandle);

    // FIXME: These should be abstracted as a encodable image buffer handle.
    auto backendParameters = ImageBuffer::backendParameters(parameters);
    auto backend = ImageBufferShareableBitmapBackend::create(backendParameters, WTFMove(handle));
    if (!backend)
        return createEmpty();
    
    auto info = ImageBuffer::populateBackendInfo<ImageBufferShareableBitmapBackend>(backendParameters);

    auto buffer = ImageBuffer::create(WTFMove(parameters), info, { }, WTFMove(backend));
    if (!buffer)
        return createEmpty();

    return WebImage::create(buffer.releaseNonNull());
}

Ref<WebImage> WebImage::create(Ref<ImageBuffer>&& buffer)
{
    return adoptRef(*new WebImage(WTFMove(buffer)));
}

WebImage::WebImage(RefPtr<ImageBuffer>&& buffer)
    : m_buffer(WTFMove(buffer))
{
}

WebImage::~WebImage() = default;

IntSize WebImage::size() const
{
    if (!m_buffer)
        return { };
    return m_buffer->backendSize();
}

const ImageBufferParameters* WebImage::parameters() const
{
    if (!m_buffer)
        return nullptr;
    return &m_buffer->parameters();
}

auto WebImage::parametersAndHandle() const -> std::optional<ParametersAndHandle>
{
    auto handle = createHandle();
    if (!handle)
        return std::nullopt;
    RELEASE_ASSERT(m_buffer);
    return { { m_buffer->parameters(), WTFMove(*handle) } };
}

GraphicsContext* WebImage::context() const
{
    if (!m_buffer)
        return nullptr;
    return &m_buffer->context();
}

RefPtr<NativeImage> WebImage::copyNativeImage(BackingStoreCopy copyBehavior) const
{
    if (!m_buffer)
        return nullptr;
    if (copyBehavior == CopyBackingStore)
        return m_buffer->copyNativeImage();
    return m_buffer->createNativeImageReference();
}

RefPtr<ShareableBitmap> WebImage::bitmap() const
{
    if (!m_buffer)
        return nullptr;
    const_cast<ImageBuffer&>(*m_buffer).flushDrawingContext();

    auto* sharing = dynamicDowncast<ImageBufferBackendHandleSharing>(m_buffer->toBackendSharing());
    return sharing ? sharing->bitmap() : nullptr;
}

#if USE(CAIRO)
RefPtr<cairo_surface_t> WebImage::createCairoSurface()
{
    if (!m_buffer)
        return nullptr;
    return m_buffer->createCairoSurface();
}
#endif

std::optional<ShareableBitmap::Handle> WebImage::createHandle(SharedMemory::Protection protection) const
{
    if (!m_buffer)
        return { };
    const_cast<ImageBuffer&>(*m_buffer).flushDrawingContext();

    auto* sharing = dynamicDowncast<ImageBufferBackendHandleSharing>(m_buffer->toBackendSharing());
    if (!sharing)
        return { };

    auto backendHandle = sharing->createBackendHandle(protection);
    if (!backendHandle)
        return { };

    return std::get<ShareableBitmap::Handle>(WTFMove(*backendHandle));
}

} // namespace WebKit
