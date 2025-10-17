/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#import "DynamicContentScalingImageBufferBackend.h"

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)

#import "Logging.h"
#import <CoreRE/RECGCommandsContext.h>
#import <WebCore/DynamicContentScalingDisplayList.h>
#import <WebCore/GraphicsContextCG.h>
#import <WebCore/PixelBuffer.h>
#import <WebCore/SharedMemory.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/MachSendRight.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/text/StringConcatenate.h>

template<> struct WTF::CFTypeTrait<CAMachPortRef> {
    static inline CFTypeID typeID(void) { return CAMachPortGetTypeID(); }
};

namespace WebKit {

using namespace WebCore;

static CFDictionaryRef makeContextOptions(const DynamicContentScalingImageBufferBackend::Parameters& parameters)
{
    RetainPtr colorSpace = parameters.colorSpace.platformColorSpace();
    if (!colorSpace)
        return nil;
    return (CFDictionaryRef)@{
        @"colorspace" : (id)colorSpace.get()
    };
}

class GraphicsContextDynamicContentScaling : public WebCore::GraphicsContextCG {
public:
    GraphicsContextDynamicContentScaling(const DynamicContentScalingImageBufferBackend::Parameters& parameters, WebCore::RenderingMode renderingMode)
        : GraphicsContextCG(adoptCF(RECGCommandsContextCreate(parameters.backendSize, makeContextOptions(parameters))).autorelease(), GraphicsContextCG::Unknown, renderingMode)
    {
    }

    bool canUseShadowBlur() const final { return false; }
};

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DynamicContentScalingImageBufferBackend);

size_t DynamicContentScalingImageBufferBackend::calculateMemoryCost(const Parameters& parameters)
{
    // FIXME: This is fairly meaningless, because we don't actually have a bitmap, and
    // should really be based on the encoded data size.
    return WebCore::ImageBufferBackend::calculateMemoryCost(parameters.backendSize, calculateBytesPerRow(parameters.backendSize));
}

std::unique_ptr<DynamicContentScalingImageBufferBackend> DynamicContentScalingImageBufferBackend::create(const Parameters& parameters, const WebCore::ImageBufferCreationContext& creationContext)
{
    if (parameters.backendSize.isEmpty())
        return nullptr;

    return std::unique_ptr<DynamicContentScalingImageBufferBackend>(new DynamicContentScalingImageBufferBackend(parameters, creationContext, WebCore::RenderingMode::Unaccelerated));
}

DynamicContentScalingImageBufferBackend::DynamicContentScalingImageBufferBackend(const Parameters& parameters, const WebCore::ImageBufferCreationContext& creationContext, WebCore::RenderingMode renderingMode)
    : ImageBufferCGBackend { parameters }
    , m_resourceCache(creationContext.dynamicContentScalingResourceCache)
    , m_renderingMode(renderingMode)
{
    // FIXME: We should make callers always specify a cache and have an assertion here instead
    // of making a temporary one. RemoteLayerWithRemoteRenderingBackingStore currently does not.
    if (!m_resourceCache)
        m_resourceCache = bridge_id_cast(adoptCF(RECGCommandsCacheCreate(nullptr)));
}

DynamicContentScalingImageBufferBackend::~DynamicContentScalingImageBufferBackend() = default;

std::optional<ImageBufferBackendHandle> DynamicContentScalingImageBufferBackend::createBackendHandle(WebCore::SharedMemory::Protection) const
{
    if (!m_context)
        return std::nullopt;

    RetainPtr<NSDictionary> options;
    RetainPtr<NSMutableArray> ports;
    if (m_resourceCache) {
        ports = adoptNS([[NSMutableArray alloc] init]);
        options = @{
            @"ports": ports.get(),
            @"cache": m_resourceCache.get()
        };
    }

    auto data = adoptCF(RECGCommandsContextCopyEncodedDataWithOptions(m_context->platformContext(), bridge_cast(options.get())));
    ASSERT(data);

    Vector<MachSendRight> sendRights;
    if (m_resourceCache) {
        sendRights = makeVector(ports.get(), [] (CFTypeRef port) -> std::optional<MachSendRight> {
            // We `create` instead of `adopt` because CAMachPort has no API to leak its reference.
            return { MachSendRight::create(CAMachPortGetPort(checked_cf_cast<CAMachPortRef>(port))) };
        });
    }

    return WebCore::DynamicContentScalingDisplayList { WebCore::SharedBuffer::create(data.get()), WTFMove(sendRights) };
}

WebCore::GraphicsContext& DynamicContentScalingImageBufferBackend::context()
{
    if (!m_context) {
        m_context = makeUnique<GraphicsContextDynamicContentScaling>(m_parameters, m_renderingMode);
        applyBaseTransform(*m_context);
    }
    return *m_context;
}

unsigned DynamicContentScalingImageBufferBackend::bytesPerRow() const
{
    return calculateBytesPerRow(m_parameters.backendSize);
}

void DynamicContentScalingImageBufferBackend::releaseGraphicsContext()
{
    m_context = nullptr;
}

bool DynamicContentScalingImageBufferBackend::canMapBackingStore() const
{
    return false;
}

RefPtr<WebCore::NativeImage> DynamicContentScalingImageBufferBackend::copyNativeImage()
{
    ASSERT_NOT_REACHED();
    return nullptr;
}

RefPtr<WebCore::NativeImage> DynamicContentScalingImageBufferBackend::createNativeImageReference()
{
    ASSERT_NOT_REACHED();
    return nullptr;
}

void DynamicContentScalingImageBufferBackend::getPixelBuffer(const WebCore::IntRect&, WebCore::PixelBuffer&)
{
    ASSERT_NOT_REACHED();
}

void DynamicContentScalingImageBufferBackend::putPixelBuffer(const WebCore::PixelBuffer&, const WebCore::IntRect&, const WebCore::IntPoint&, WebCore::AlphaPremultiplication)
{
    ASSERT_NOT_REACHED();
}

String DynamicContentScalingImageBufferBackend::debugDescription() const
{
    TextStream stream;
    stream << "DynamicContentScalingImageBufferBackend " << this;
    return stream.release();
}

}

#endif // ENABLE(RE_DYNAMIC_CONTENT_SCALING)
