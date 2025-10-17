/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#include "WebGPUCompositorIntegrationImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include "WebGPUDevice.h"
#include "WebGPUQueue.h"
#include "WebGPUTextureFormat.h"
#include <CoreFoundation/CoreFoundation.h>
#include <WebCore/IOSurface.h>
#include <WebCore/NativeImage.h>
#include <WebGPU/WebGPUExt.h>
#include <pal/spi/cg/CoreGraphicsSPI.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/spi/cocoa/IOSurfaceSPI.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CompositorIntegrationImpl);

CompositorIntegrationImpl::CompositorIntegrationImpl(ConvertToBackingContext& convertToBackingContext)
    : m_convertToBackingContext(convertToBackingContext)
{
}

CompositorIntegrationImpl::~CompositorIntegrationImpl() = default;

void CompositorIntegrationImpl::prepareForDisplay(uint32_t frameIndex, CompletionHandler<void()>&& completionHandler)
{
    if (auto* presentationContext = m_presentationContext.get())
        presentationContext->present(frameIndex);

    m_onSubmittedWorkScheduledCallback(WTFMove(completionHandler));
}

#if PLATFORM(COCOA)
Vector<MachSendRight> CompositorIntegrationImpl::recreateRenderBuffers(int width, int height, WebCore::DestinationColorSpace&& colorSpace, WebCore::AlphaPremultiplication alphaMode, TextureFormat textureFormat, Device& device)
{
    m_renderBuffers.clear();
    m_device = device;

    if (auto* presentationContext = m_presentationContext.get()) {
        static_cast<PresentationContext*>(presentationContext)->unconfigure();
        presentationContext->setSize(width, height);
    }

    constexpr int max2DTextureSize = 16384;
    width = std::max(1, std::min(max2DTextureSize, width));
    height = std::max(1, std::min(max2DTextureSize, height));
    IOSurface::Format colorFormat;
    switch (textureFormat) {
    case TextureFormat::Rgba8unorm:
    case TextureFormat::Rgba8unormSRGB:
        colorFormat = alphaMode == AlphaPremultiplication::Unpremultiplied ? IOSurface::Format::RGBX : IOSurface::Format::RGBA;
        break;
#if HAVE(HDR_SUPPORT)
    case TextureFormat::Rgba16float:
        colorFormat = IOSurface::Format::RGBA16F;
        break;
#endif
    default:
        colorFormat = alphaMode == AlphaPremultiplication::Unpremultiplied ? IOSurface::Format::BGRX : IOSurface::Format::BGRA;
        break;
    }

    if (auto buffer = WebCore::IOSurface::create(nullptr, WebCore::IntSize(width, height), colorSpace, IOSurface::Name::WebGPU, colorFormat))
        m_renderBuffers.append(makeUniqueRefFromNonNullUniquePtr(WTFMove(buffer)));
    if (auto buffer = WebCore::IOSurface::create(nullptr, WebCore::IntSize(width, height), colorSpace, IOSurface::Name::WebGPU, colorFormat))
        m_renderBuffers.append(makeUniqueRefFromNonNullUniquePtr(WTFMove(buffer)));
    if (auto buffer = WebCore::IOSurface::create(nullptr, WebCore::IntSize(width, height), colorSpace, IOSurface::Name::WebGPU, colorFormat))
        m_renderBuffers.append(makeUniqueRefFromNonNullUniquePtr(WTFMove(buffer)));

    {
        auto renderBuffers = adoptCF(CFArrayCreateMutable(kCFAllocatorDefault, m_renderBuffers.size(), &kCFTypeArrayCallBacks));
        for (auto& ioSurface : m_renderBuffers)
            CFArrayAppendValue(renderBuffers.get(), ioSurface->surface());
        m_renderBuffersWereRecreatedCallback(static_cast<CFArrayRef>(renderBuffers));
    }

    return m_renderBuffers.map([](const auto& renderBuffer) {
        return renderBuffer->createSendRight();
    });
}
#endif

void CompositorIntegrationImpl::withDisplayBufferAsNativeImage(uint32_t bufferIndex, Function<void(WebCore::NativeImage*)> completion)
{
    if (!m_renderBuffers.size() || bufferIndex >= m_renderBuffers.size() || !m_device.get())
        return completion(nullptr);

    RefPtr<NativeImage> displayImage;
    bool isIOSurfaceSupportedFormat = false;
    if (auto* presentationContextPtr = m_presentationContext.get())
        displayImage = presentationContextPtr->getMetalTextureAsNativeImage(bufferIndex, isIOSurfaceSupportedFormat);

    if (!displayImage) {
        if (!isIOSurfaceSupportedFormat)
            return completion(nullptr);

        auto& renderBuffer = m_renderBuffers[bufferIndex];
        RetainPtr<CGContextRef> cgContext = renderBuffer->createPlatformContext();
        if (cgContext)
            displayImage = NativeImage::create(renderBuffer->createImage(cgContext.get()));
    }

    if (!displayImage)
        return completion(nullptr);

    CGImageSetCachingFlags(displayImage->platformImage().get(), kCGImageCachingTransient);
    completion(displayImage.get());
}

void CompositorIntegrationImpl::paintCompositedResultsToCanvas(WebCore::ImageBuffer&, uint32_t)
{
    ASSERT_NOT_REACHED();
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
