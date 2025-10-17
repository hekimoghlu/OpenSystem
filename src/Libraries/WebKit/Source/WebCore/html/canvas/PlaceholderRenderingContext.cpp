/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#include "PlaceholderRenderingContext.h"

#if ENABLE(OFFSCREEN_CANVAS)

#include "GraphicsLayerContentsDisplayDelegate.h"
#include "HTMLCanvasElement.h"
#include "OffscreenCanvas.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PlaceholderRenderingContextSource);

Ref<PlaceholderRenderingContextSource> PlaceholderRenderingContextSource::create(PlaceholderRenderingContext& context)
{
    return adoptRef(*new PlaceholderRenderingContextSource(context));
}

PlaceholderRenderingContextSource::PlaceholderRenderingContextSource(PlaceholderRenderingContext& placeholder)
    : m_placeholder(placeholder)
{
}

void PlaceholderRenderingContextSource::setPlaceholderBuffer(ImageBuffer& imageBuffer)
{
    {
        Locker locker { m_lock };
        if (m_delegate)
            m_delegate->tryCopyToLayer(imageBuffer);
    }

    RefPtr clone = imageBuffer.clone();
    if (!clone)
        return;
    std::unique_ptr serializedClone = ImageBuffer::sinkIntoSerializedImageBuffer(WTFMove(clone));
    if (!serializedClone)
        return;
    callOnMainThread([weakPlaceholder = m_placeholder, buffer = WTFMove(serializedClone)] () mutable {
        RefPtr placeholder = weakPlaceholder.get();
        if (!placeholder)
            return;
        RefPtr imageBuffer = SerializedImageBuffer::sinkIntoImageBuffer(WTFMove(buffer), placeholder->canvas().scriptExecutionContext()->graphicsClient());
        if (!imageBuffer)
            return;
        placeholder->setPlaceholderBuffer(imageBuffer.releaseNonNull());
    });
}

void PlaceholderRenderingContextSource::setContentsToLayer(GraphicsLayer& layer)
{
    Locker locker { m_lock };
    m_delegate = layer.createAsyncContentsDisplayDelegate(m_delegate.get());
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PlaceholderRenderingContext);

std::unique_ptr<PlaceholderRenderingContext> PlaceholderRenderingContext::create(HTMLCanvasElement& element)
{
    return std::unique_ptr<PlaceholderRenderingContext> { new PlaceholderRenderingContext(element) };
}

PlaceholderRenderingContext::PlaceholderRenderingContext(HTMLCanvasElement& canvas)
    : CanvasRenderingContext(canvas, Type::Placeholder)
    , m_source(PlaceholderRenderingContextSource::create(*this))
{
}

HTMLCanvasElement& PlaceholderRenderingContext::canvas() const
{
    return static_cast<HTMLCanvasElement&>(canvasBase());
}

IntSize PlaceholderRenderingContext::size() const
{
    return canvas().size();
}

void PlaceholderRenderingContext::setContentsToLayer(GraphicsLayer& layer)
{
    m_source->setContentsToLayer(layer);
}

void PlaceholderRenderingContext::setPlaceholderBuffer(Ref<ImageBuffer>&& buffer)
{
    canvasBase().setImageBufferAndMarkDirty(WTFMove(buffer));
}

}

#endif
