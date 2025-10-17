/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#include "RenderImageResourceStyleImage.h"

#include "CachedImage.h"
#include "RenderElement.h"
#include "RenderStyleInlines.h"
#include "StyleCachedImage.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderImageResourceStyleImage);

RenderImageResourceStyleImage::RenderImageResourceStyleImage(StyleImage& styleImage)
    : m_styleImage(styleImage)
{
}

void RenderImageResourceStyleImage::initialize(RenderElement& renderer)
{
    RenderImageResource::initialize(renderer, m_styleImage->hasCachedImage() ? m_styleImage.get().cachedImage() : nullptr);
    m_styleImage->addClient(renderer);
}

void RenderImageResourceStyleImage::shutdown()
{
    RenderImageResource::shutdown();
    if (auto renderer = this->renderer())
        m_styleImage->removeClient(*renderer);
}

RefPtr<Image> RenderImageResourceStyleImage::image(const IntSize& size) const
{
    // Generated content may trigger calls to image() while we're still pending, don't assert but gracefully exit.
    if (m_styleImage->isPending())
        return &Image::nullImage();
    if (auto image = m_styleImage->image(renderer(), size))
        return image;
    return &Image::nullImage();
}

void RenderImageResourceStyleImage::setContainerContext(const IntSize& size, const URL&)
{
    if (auto renderer = this->renderer())
        m_styleImage->setContainerContextForRenderer(*renderer, size, renderer->style().usedZoom());
}

} // namespace WebCore
