/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
#import "GraphicsLayerAsyncContentsDisplayDelegateCocoa.h"

#import "GraphicsLayerCA.h"
#import "ImageBuffer.h"
#import "NativeImage.h"
#import "WebCoreCALayerExtras.h"

namespace WebCore {

GraphicsLayerAsyncContentsDisplayDelegateCocoa::GraphicsLayerAsyncContentsDisplayDelegateCocoa(GraphicsLayerCA& layer)
{
    m_layer = adoptNS([[CALayer alloc] init]);
    [m_layer setName:@"OffscreenCanvasLayer"];

    layer.setContentsToPlatformLayer(m_layer.get(), GraphicsLayer::ContentsLayerPurpose::Canvas);
}

bool GraphicsLayerAsyncContentsDisplayDelegateCocoa::tryCopyToLayer(ImageBuffer& image)
{
    m_image = ImageBuffer::sinkIntoNativeImage(image.clone());
    if (!m_image)
        return false;

    [CATransaction begin];
    [CATransaction setDisableActions:YES];

    [m_layer setContents:(__bridge id)m_image->platformImage().get()];

    [CATransaction commit];

    return true;
}

void GraphicsLayerAsyncContentsDisplayDelegateCocoa::updateGraphicsLayerCA(GraphicsLayerCA& layer)
{
    layer.setContentsToPlatformLayer(m_layer.get(), GraphicsLayer::ContentsLayerPurpose::Canvas);
    if (m_image)
        [m_layer setContents:(__bridge id)m_image->platformImage().get()];
}

}
