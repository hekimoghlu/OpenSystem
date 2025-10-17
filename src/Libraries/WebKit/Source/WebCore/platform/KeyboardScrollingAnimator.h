/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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

#include "KeyboardEvent.h"
#include "KeyboardScroll.h" // FIXME: This is a layering violation.
#include "RectEdges.h"
#include "ScrollableArea.h"
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class PlatformKeyboardEvent;

enum class KeyboardScrollingKey : uint8_t {
    LeftArrow,
    RightArrow,
    UpArrow,
    DownArrow,
    Space,
    PageUp,
    PageDown,
    Home,
    End
};

const std::optional<KeyboardScrollingKey> keyboardScrollingKeyForKeyboardEvent(const KeyboardEvent&);
const std::optional<ScrollDirection> scrollDirectionForKeyboardEvent(const KeyboardEvent&);
const std::optional<ScrollGranularity> scrollGranularityForKeyboardEvent(const KeyboardEvent&);

class KeyboardScrollingAnimator final : public CanMakeWeakPtr<KeyboardScrollingAnimator>, public CanMakeCheckedPtr<KeyboardScrollingAnimator> {
    WTF_MAKE_TZONE_ALLOCATED(KeyboardScrollingAnimator);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(KeyboardScrollingAnimator);
    WTF_MAKE_NONCOPYABLE(KeyboardScrollingAnimator);
public:
    KeyboardScrollingAnimator(ScrollableArea&);

    WEBCORE_EXPORT bool beginKeyboardScrollGesture(ScrollDirection, ScrollGranularity, bool isKeyRepeat);
    WEBCORE_EXPORT void handleKeyUpEvent();
    WEBCORE_EXPORT void stopScrollingImmediately();

private:
    std::optional<KeyboardScroll> makeKeyboardScroll(ScrollDirection, ScrollGranularity) const;
    float scrollDistance(ScrollDirection, ScrollGranularity) const;
    RectEdges<bool> scrollingDirections() const;

    ScrollableArea& m_scrollableArea;
    bool m_scrollTriggeringKeyIsPressed { false };
};

} // namespace WebCore
