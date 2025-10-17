/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
#ifndef LocalWindowsContext_h
#define LocalWindowsContext_h

#include "GraphicsContext.h"

namespace WebCore {

class LocalWindowsContext {
    WTF_MAKE_NONCOPYABLE(LocalWindowsContext);
public:
    LocalWindowsContext(GraphicsContext& graphicsContext, const IntRect& rect, bool supportAlphaBlend = true)
        : m_graphicsContext(graphicsContext)
        , m_rect(rect)
        , m_supportAlphaBlend(supportAlphaBlend)
    {
        m_hdc = m_graphicsContext.getWindowsContext(m_rect, m_supportAlphaBlend);
    }

    ~LocalWindowsContext()
    {
        if (m_hdc)
            m_graphicsContext.releaseWindowsContext(m_hdc, m_rect, m_supportAlphaBlend);
    }

    HDC hdc() const { return m_hdc; }

private:
    GraphicsContext& m_graphicsContext;
    HDC m_hdc;
    IntRect m_rect;
    bool m_supportAlphaBlend;
};

} // namespace WebCore
#endif // LocalWindowsContext_h
