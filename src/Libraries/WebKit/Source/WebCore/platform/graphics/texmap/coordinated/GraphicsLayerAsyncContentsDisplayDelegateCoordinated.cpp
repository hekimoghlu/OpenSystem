/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#include "GraphicsLayerAsyncContentsDisplayDelegateCoordinated.h"

#if USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayerBufferNativeImage.h"
#include "GraphicsLayer.h"
#include "GraphicsLayerContentsDisplayDelegateCoordinated.h"
#include "ImageBuffer.h"
#include "NativeImage.h"
#include "TextureMapperFlags.h"

namespace WebCore {

GraphicsLayerAsyncContentsDisplayDelegateCoordinated::GraphicsLayerAsyncContentsDisplayDelegateCoordinated(GraphicsLayer& layer)
    : m_delegate(GraphicsLayerContentsDisplayDelegateCoordinated::create())
{
    layer.setContentsDisplayDelegate(m_delegate.ptr(), GraphicsLayer::ContentsLayerPurpose::Canvas);
}

GraphicsLayerAsyncContentsDisplayDelegateCoordinated::~GraphicsLayerAsyncContentsDisplayDelegateCoordinated() = default;

bool GraphicsLayerAsyncContentsDisplayDelegateCoordinated::tryCopyToLayer(ImageBuffer& imageBuffer)
{
    auto image = ImageBuffer::sinkIntoNativeImage(imageBuffer.clone());
    if (!image)
        return false;

    m_delegate->setDisplayBuffer(CoordinatedPlatformLayerBufferNativeImage::create(image.releaseNonNull(), nullptr));
    return true;
}

void GraphicsLayerAsyncContentsDisplayDelegateCoordinated::updateGraphicsLayer(GraphicsLayer& layer)
{
    layer.setContentsDisplayDelegate(m_delegate.ptr(), GraphicsLayer::ContentsLayerPurpose::Canvas);
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
