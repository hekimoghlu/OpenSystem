/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "CanvasRenderingContext.h"

#include "CachedImage.h"
#include "CanvasPattern.h"
#include "DestinationColorSpace.h"
#include "GraphicsLayer.h"
#include "GraphicsLayerContentsDisplayDelegate.h"
#include "HTMLCanvasElement.h"
#include "HTMLImageElement.h"
#include "HTMLVideoElement.h"
#include "Image.h"
#include "ImageBitmap.h"
#include "OriginAccessPatterns.h"
#include "PixelFormat.h"
#include "SVGImageElement.h"
#include "SecurityOrigin.h"
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

#if USE(SKIA)
#include "CanvasRenderingContext2DBase.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CanvasRenderingContext);

Lock CanvasRenderingContext::s_instancesLock;

UncheckedKeyHashSet<CanvasRenderingContext*>& CanvasRenderingContext::instances()
{
    static NeverDestroyed<UncheckedKeyHashSet<CanvasRenderingContext*>> instances;
    return instances;
}

Lock& CanvasRenderingContext::instancesLock()
{
    return s_instancesLock;
}

CanvasRenderingContext::CanvasRenderingContext(CanvasBase& canvas, Type type)
    : m_canvas(canvas)
    , m_type(type)
{
    Locker locker { instancesLock() };
    instances().add(this);
}

CanvasRenderingContext::~CanvasRenderingContext()
{
    Locker locker { instancesLock() };
    ASSERT(instances().contains(this));
    instances().remove(this);
}

void CanvasRenderingContext::ref() const
{
    m_canvas->ref();
}

void CanvasRenderingContext::deref() const
{
    m_canvas->deref();
}

RefPtr<ImageBuffer> CanvasRenderingContext::surfaceBufferToImageBuffer(SurfaceBuffer)
{
    // This will be removed once all contexts store their own buffers.
    return canvasBase().buffer();
}

bool CanvasRenderingContext::isSurfaceBufferTransparentBlack(SurfaceBuffer) const
{
    return false;
}

bool CanvasRenderingContext::delegatesDisplay() const
{
#if USE(SKIA)
    if (auto* context2D = dynamicDowncast<CanvasRenderingContext2DBase>(*this))
        return context2D->isAccelerated();
#endif
    return isPlaceholder() || isGPUBased();
}

RefPtr<GraphicsLayerContentsDisplayDelegate> CanvasRenderingContext::layerContentsDisplayDelegate()
{
    return nullptr;
}

void CanvasRenderingContext::setContentsToLayer(GraphicsLayer& layer)
{
    layer.setContentsDisplayDelegate(layerContentsDisplayDelegate(), GraphicsLayer::ContentsLayerPurpose::Canvas);
}

RefPtr<ImageBuffer> CanvasRenderingContext::transferToImageBuffer()
{
    ASSERT_NOT_REACHED(); // Implemented and called only for offscreen capable contexts.
    return nullptr;
}

ImageBufferPixelFormat CanvasRenderingContext::pixelFormat() const
{
    return ImageBufferPixelFormat::BGRA8;
}

DestinationColorSpace CanvasRenderingContext::colorSpace() const
{
    return DestinationColorSpace::SRGB();
}

bool CanvasRenderingContext::willReadFrequently() const
{
    return false;
}

bool CanvasRenderingContext::taintsOrigin(const CanvasPattern* pattern)
{
    return pattern && !pattern->originClean();
}

bool CanvasRenderingContext::taintsOrigin(const CanvasBase* sourceCanvas)
{
    return sourceCanvas && !sourceCanvas->originClean();
}

bool CanvasRenderingContext::taintsOrigin(const CachedImage* cachedImage)
{
    if (!cachedImage)
        return false;

    RefPtr image = cachedImage->image();
    if (!image)
        return false;

    if (image->sourceURL().protocolIsData())
        return false;

    if (image->renderingTaintsOrigin())
        return true;

    if (cachedImage->isCORSCrossOrigin())
        return true;

    ASSERT(m_canvas->securityOrigin());
    ASSERT(cachedImage->origin());
    ASSERT(m_canvas->securityOrigin()->toString() == cachedImage->origin()->toString());
    return false;
}

bool CanvasRenderingContext::taintsOrigin(const HTMLImageElement* element)
{
    return element && taintsOrigin(element->cachedImage());
}

bool CanvasRenderingContext::taintsOrigin(const SVGImageElement* element)
{
    return element && taintsOrigin(element->cachedImage());
}

bool CanvasRenderingContext::taintsOrigin(const HTMLVideoElement* video)
{
#if ENABLE(VIDEO)
    return video && video->taintsOrigin(*m_canvas->securityOrigin());
#else
    UNUSED_PARAM(video);
    return false;
#endif
}

bool CanvasRenderingContext::taintsOrigin(const ImageBitmap* imageBitmap)
{
    return imageBitmap && !imageBitmap->originClean();
}

bool CanvasRenderingContext::taintsOrigin(const URL& url)
{
    return !url.protocolIsData() && !m_canvas->securityOrigin()->canRequest(url, OriginAccessPatternsForWebProcess::singleton());
}

void CanvasRenderingContext::checkOrigin(const URL& url)
{
    if (m_canvas->originClean() && taintsOrigin(url))
        m_canvas->setOriginTainted();
}

void CanvasRenderingContext::checkOrigin(const CSSStyleImageValue&)
{
    m_canvas->setOriginTainted();
}

} // namespace WebCore
