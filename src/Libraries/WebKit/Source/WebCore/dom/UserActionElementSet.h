/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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

#include <wtf/Forward.h>
#include <wtf/OptionSet.h>
#include <wtf/Ref.h>
#include <wtf/WeakHashMap.h>

namespace WebCore {

class Element;
class WeakPtrImplWithEventTargetData;

class UserActionElementSet {
public:
    bool isActive(const Element& element) { return hasFlag(element, Flag::IsActive); }
    bool isFocused(const Element& element) { return hasFlag(element, Flag::IsFocused); }
    bool isHovered(const Element& element) { return hasFlag(element, Flag::IsHovered); }
    bool isInActiveChain(const Element& element) { return hasFlag(element, Flag::InActiveChain); }
    bool isBeingDragged(const Element& element) { return hasFlag(element, Flag::IsBeingDragged); }
    bool hasFocusVisible(const Element& element) { return hasFlag(element, Flag::HasFocusVisible); }
    bool hasFocusWithin(const Element& element) { return hasFlag(element, Flag::HasFocusWithin); }

    void setActive(Element& element, bool enable) { setFlags(element, enable, Flag::IsActive); }
    void setFocused(Element& element, bool enable) { setFlags(element, enable, Flag::IsFocused); }
    void setHovered(Element& element, bool enable) { setFlags(element, enable, Flag::IsHovered); }
    void setInActiveChain(Element& element, bool enable) { setFlags(element, enable, Flag::InActiveChain); }
    void setBeingDragged(Element& element, bool enable) { setFlags(element, enable, Flag::IsBeingDragged); }
    void setHasFocusVisible(Element& element, bool enable) { setFlags(element, enable, Flag::HasFocusVisible); }
    void setHasFocusWithin(Element& element, bool enable) { setFlags(element, enable, Flag::HasFocusWithin); }

    void clearActiveAndHovered(Element& element) { clearFlags(element, { Flag::IsActive, Flag::InActiveChain, Flag::IsHovered }); }
    void clearAllForElement(Element& element) { clearFlags(element, { Flag::IsActive, Flag::InActiveChain, Flag::IsHovered, Flag::IsFocused, Flag::IsBeingDragged, Flag::HasFocusVisible, Flag::HasFocusWithin }); }

    void clear();

private:
    enum class Flag {
        IsActive        = 1 << 0,
        InActiveChain   = 1 << 1,
        IsHovered       = 1 << 2,
        IsFocused       = 1 << 3,
        IsBeingDragged  = 1 << 4,
        HasFocusVisible = 1 << 5,
        HasFocusWithin  = 1 << 6,
    };

    void setFlags(Element& element, bool enable, OptionSet<Flag> flags) { enable ? setFlags(element, flags) : clearFlags(element, flags); }
    void setFlags(Element&, OptionSet<Flag>);
    void clearFlags(Element&, OptionSet<Flag>);
    bool hasFlag(const Element&, Flag) const;

    WeakHashMap<Element, OptionSet<Flag>, WeakPtrImplWithEventTargetData> m_elements;
};

} // namespace WebCore
