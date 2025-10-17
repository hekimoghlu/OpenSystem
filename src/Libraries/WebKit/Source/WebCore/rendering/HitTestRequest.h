/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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

#include "HitTestSource.h"
#include <wtf/Assertions.h>
#include <wtf/OptionSet.h>

namespace WebCore {

class HitTestRequest {
public:
    enum class Type {
        ReadOnly = 1 << 0,
        Active = 1 << 1,
        Move = 1 << 2,
        Release = 1 << 3,
        IgnoreCSSPointerEventsProperty = 1 << 4,
        IgnoreClipping = 1 << 5,
        SVGClipContent = 1 << 6,
        TouchEvent = 1 << 7,
        DisallowUserAgentShadowContent = 1 << 8,
        DisallowUserAgentShadowContentExceptForImageOverlays = 1 << 9,
        AllowFrameScrollbars = 1 << 10,
        AllowChildFrameContent = 1 << 11,
        AllowVisibleChildFrameContentOnly = 1 << 12,
        ChildFrameHitTest = 1 << 13,
        AccessibilityHitTest = 1 << 14,
        // Collect a list of nodes instead of just one. Used for elementsFromPoint and rect-based tests.
        CollectMultipleElements = 1 << 15,
        // When using list-based testing, continue hit testing even after a hit has been found.
        IncludeAllElementsUnderPoint = 1 << 16,
        PenEvent = 1 << 17,
    };

    static constexpr OptionSet defaultTypes = { Type::ReadOnly, Type::Active, Type::DisallowUserAgentShadowContent };

    static inline void assertConsistentType(OptionSet<Type> type)
    {
#if ASSERT_ENABLED
        ASSERT(!type.containsAll({ Type::DisallowUserAgentShadowContentExceptForImageOverlays, Type::DisallowUserAgentShadowContent }));
        ASSERT_IMPLIES(type.contains(Type::IncludeAllElementsUnderPoint), type.contains(Type::CollectMultipleElements));
#else
        UNUSED_PARAM(type);
#endif
    }

    HitTestRequest(HitTestSource source, OptionSet<Type> type = defaultTypes)
        : m_type { type }
        , m_source { source }
    {
        assertConsistentType(type);
    }

    // FIXME: This constructor should be phased out in favor of the `HitTestSource` version above, such that all call sites must
    // consider whether the hit test request is user-triggered or bindings-triggered.
    HitTestRequest(OptionSet<Type> type = defaultTypes)
        : m_type { type }
    {
        assertConsistentType(type);
    }

    bool readOnly() const { return m_type.contains(Type::ReadOnly); }
    bool active() const { return m_type.contains(Type::Active); }
    bool move() const { return m_type.contains(Type::Move); }
    bool release() const { return m_type.contains(Type::Release); }
    bool ignoreCSSPointerEventsProperty() const { return m_type.contains(Type::IgnoreCSSPointerEventsProperty); }
    bool ignoreClipping() const { return m_type.contains(Type::IgnoreClipping); }
    bool svgClipContent() const { return m_type.contains(Type::SVGClipContent); }
    bool touchEvent() const { return m_type.contains(Type::TouchEvent); }
    bool mouseEvent() const { return !touchEvent() && !penEvent(); }
    bool penEvent() const { return m_type.contains(Type::PenEvent); }
    bool disallowsUserAgentShadowContent() const { return m_type.contains(Type::DisallowUserAgentShadowContent); }
    bool disallowsUserAgentShadowContentExceptForImageOverlays() const { return m_type.contains(Type::DisallowUserAgentShadowContentExceptForImageOverlays); }
    bool allowsFrameScrollbars() const { return m_type.contains(Type::AllowFrameScrollbars); }
    bool allowsChildFrameContent() const { return m_type.contains(Type::AllowChildFrameContent); }
    bool allowsVisibleChildFrameContent() const { return m_type.contains(Type::AllowVisibleChildFrameContentOnly); }
    bool allowsAnyFrameContent() const { return allowsChildFrameContent() ||  allowsVisibleChildFrameContent(); }
    bool isChildFrameHitTest() const { return m_type.contains(Type::ChildFrameHitTest); }
    bool resultIsElementList() const { return m_type.contains(Type::CollectMultipleElements); }
    bool includesAllElementsUnderPoint() const { return m_type.contains(Type::IncludeAllElementsUnderPoint); }
    bool userTriggered() const { return m_source == HitTestSource::User; }

    // Convenience functions
    bool touchMove() const { return move() && touchEvent(); }
    bool touchRelease() const { return release() && touchEvent(); }

    OptionSet<Type> type() const { return m_type; }

private:
    OptionSet<Type> m_type;
    HitTestSource m_source { HitTestSource::User };
};

} // namespace WebCore
