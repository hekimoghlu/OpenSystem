/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#pragma once

#if USE(GRAPHICS_LAYER_WC) && ENABLE(WEBGL)

#include "GraphicsContextGLIdentifier.h"
#include "WCContentBufferIdentifier.h"
#include <WebCore/WCPlatformLayer.h>

namespace WebKit {

class WCPlatformLayerGCGL : public WebCore::WCPlatformLayer {
public:
    Vector<WCContentBufferIdentifier> takeContentBufferIdentifiers()
    {
        return std::exchange(m_contentBufferIdentifiers, { });
    }

    void addContentBufferIdentifier(WCContentBufferIdentifier contentBuffer)
    {
        m_contentBufferIdentifiers.append(contentBuffer);
        // FIXME: TextureMapperGCGLPlatformLayer doesn't support double buffering yet.
        ASSERT(m_contentBufferIdentifiers.size() == 1);
    }

private:
    Vector<WCContentBufferIdentifier> m_contentBufferIdentifiers;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC) && ENABLE(WEBGL)
