/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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

#include "DocumentMarker.h"
#include <wtf/Markable.h>
#include <wtf/Vector.h>
#include <wtf/WallTime.h>

namespace WebCore {
class RenderedDocumentMarker;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::RenderedDocumentMarker> : std::true_type { };
}

namespace WebCore {

class RenderedDocumentMarker : public DocumentMarker {
public:
    explicit RenderedDocumentMarker(DocumentMarker&& marker)
        : DocumentMarker(WTFMove(marker))
    {
    }

    bool contains(const FloatPoint& point) const
    {
        ASSERT(m_isValid);
        for (const auto& rect : m_rects) {
            if (rect.contains(point))
                return true;
        }
        return false;
    }

    void setUnclippedAbsoluteRects(Vector<FloatRect>&& rects)
    {
        m_isValid = true;
        m_rects = WTFMove(rects);
    }

    const Vector<FloatRect, 1>& unclippedAbsoluteRects() const
    {
        ASSERT(m_isValid);
        return m_rects;
    }

    void invalidate()
    {
        m_isValid = false;
        m_rects.clear();
    }

    bool isValid() const { return m_isValid; }

    float opacity() const { return m_opacity; }
    void setOpacity(float opacity) { m_opacity = opacity; }

    bool isBeingDismissed() const { return m_isBeingDismissed; }

    void setBeingDismissed(bool beingDismissed)
    {
        if (m_isBeingDismissed == beingDismissed)
            return;

        m_isBeingDismissed = beingDismissed;
        if (m_isBeingDismissed)
            m_animationStartTime = WallTime::now();
        else
            m_animationStartTime.reset();
    }

    Markable<WallTime> animationStartTime() const { return m_animationStartTime; }

private:
    Vector<FloatRect, 1> m_rects;
    bool m_isValid { false };
    bool m_isBeingDismissed { false };

    float m_opacity { 1.0 };
    Markable<WallTime> m_animationStartTime;
};

} // namespace WebCore
