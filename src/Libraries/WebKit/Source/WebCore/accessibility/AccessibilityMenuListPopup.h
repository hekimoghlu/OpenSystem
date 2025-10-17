/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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

#include "AccessibilityMockObject.h"

namespace WebCore {

class AccessibilityMenuList;
class AccessibilityMenuListOption;
class HTMLElement;

class AccessibilityMenuListPopup final : public AccessibilityMockObject {
    friend class AXObjectCache;
public:
    static Ref<AccessibilityMenuListPopup> create(AXID axID) { return adoptRef(*new AccessibilityMenuListPopup(axID)); }

    bool isEnabled() const final;
    bool isOffScreen() const final;

    void didUpdateActiveOption(int optionIndex);

private:
    explicit AccessibilityMenuListPopup(AXID);

    bool isMenuListPopup() const final { return true; }

    LayoutRect elementRect() const final { return LayoutRect(); }
    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::MenuListPopup; }

    bool isVisible() const final;
    bool press() final;
    void addChildren() final;
    void handleChildrenChanged();
    bool computeIsIgnored() const final;

    AccessibilityMenuListOption* menuListOptionAccessibilityObject(HTMLElement*) const;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityMenuListPopup)
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isMenuListPopup(); }
    static bool isType(const WebCore::AXCoreObject& object)
    {
        auto* accessibilityObject = dynamicDowncast<WebCore::AccessibilityObject>(object);
        return accessibilityObject && accessibilityObject->isMenuListPopup();
    }
SPECIALIZE_TYPE_TRAITS_END()
