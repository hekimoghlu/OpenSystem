/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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

#include "FloatPoint.h"
#include "ScrollTypes.h"
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FloatPoint;
class ScrollAnimation;
struct ScrollExtents;

class ScrollAnimationClient {
public:
    virtual ~ScrollAnimationClient() = default;

    virtual void scrollAnimationDidUpdate(ScrollAnimation&, const FloatPoint& /* currentOffset */) { }
    virtual void scrollAnimationWillStart(ScrollAnimation&) { }
    virtual void scrollAnimationDidEnd(ScrollAnimation&) { }
    virtual ScrollExtents scrollExtentsForAnimation(ScrollAnimation&) = 0;
    virtual FloatSize overscrollAmount(ScrollAnimation&) = 0;
    virtual FloatPoint scrollOffset(ScrollAnimation&) = 0;
};

class ScrollAnimation {
    WTF_MAKE_TZONE_ALLOCATED(ScrollAnimation);
public:
    enum class Type {
        Smooth,
        Kinetic,
        Momentum,
        RubberBand,
        Keyboard,
    };

    ScrollAnimation(Type animationType, ScrollAnimationClient& client)
        : m_client(client)
        , m_animationType(animationType)
    { }
    virtual ~ScrollAnimation() = default;
    
    Type type() const { return m_animationType; }

    virtual ScrollClamping clamping() const { return ScrollClamping::Clamped; }

    virtual bool retargetActiveAnimation(const FloatPoint& newDestinationOffset) = 0;
    virtual void stop()
    {
        if (!m_isActive)
            return;
        didEnd();
    }
    virtual bool isActive() const { return m_isActive; }
    virtual void updateScrollExtents() { };
    
    FloatPoint currentOffset() const { return m_currentOffset; }
    virtual std::optional<FloatPoint> destinationOffset() const { return std::nullopt; }

    virtual void serviceAnimation(MonotonicTime) = 0;

    virtual String debugDescription() const = 0;

protected:
    void didStart(MonotonicTime currentTime)
    {
        m_startTime = currentTime;
        m_isActive = true;
        m_client.scrollAnimationWillStart(*this);
    }
    
    void didEnd()
    {
        m_isActive = false;
        m_client.scrollAnimationDidEnd(*this);
    }
    
    Seconds timeSinceStart(MonotonicTime currentTime) const
    {
        return currentTime - m_startTime;
    }

    ScrollAnimationClient& m_client;
    const Type m_animationType;
    bool m_isActive { false };
    MonotonicTime m_startTime;
    FloatPoint m_currentOffset;
};

WTF::TextStream& operator<<(WTF::TextStream&, ScrollAnimation::Type);
WTF::TextStream& operator<<(WTF::TextStream&, const ScrollAnimation&);

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_SCROLL_ANIMATION(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(ToValueTypeName) \
    static bool isType(const WebCore::ScrollAnimation& scrollAnimation) { return scrollAnimation.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
