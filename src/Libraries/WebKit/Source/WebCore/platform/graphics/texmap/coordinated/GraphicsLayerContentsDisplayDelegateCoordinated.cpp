/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 8, 2024.
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
#include "GraphicsLayerContentsDisplayDelegateCoordinated.h"

#if USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "CoordinatedPlatformLayerBuffer.h"

namespace WebCore {

GraphicsLayerContentsDisplayDelegateCoordinated::GraphicsLayerContentsDisplayDelegateCoordinated() = default;

GraphicsLayerContentsDisplayDelegateCoordinated::~GraphicsLayerContentsDisplayDelegateCoordinated() = default;

void GraphicsLayerContentsDisplayDelegateCoordinated::setDisplayBuffer(std::unique_ptr<CoordinatedPlatformLayerBuffer>&& displayBuffer)
{
    m_displayBuffer = WTFMove(displayBuffer);
}

bool GraphicsLayerContentsDisplayDelegateCoordinated::display(CoordinatedPlatformLayer& layer)
{
    if (!m_displayBuffer)
        return false;

    layer.setContentsBuffer(WTFMove(m_displayBuffer));
    return true;
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
