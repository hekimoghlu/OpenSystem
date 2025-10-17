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

#include "AccessibilityRenderObject.h"

namespace WebCore {

class AccessibilityMenuListPopup;
class RenderMenuList;

class AccessibilityMenuList final : public AccessibilityRenderObject {
public:
    static Ref<AccessibilityMenuList> create(AXID, RenderMenuList&, AXObjectCache&);

    bool isCollapsed() const final;
    bool press() final;

    void didUpdateActiveOption(int optionIndex);

private:
    explicit AccessibilityMenuList(AXID, RenderMenuList&, AXObjectCache&);

    bool isMenuList() const final { return true; }
    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::PopUpButton; }

    bool canSetFocusAttribute() const final;
    void addChildren() final;
    void updateChildrenIfNecessary() final;
    // This class' children are initialized once in the constructor with m_popup.
    void clearChildren() final { };
    void setNeedsToUpdateChildren() final { };

    // FIXME: Nothing calls AXObjectCache::remove for m_popup.
    Ref<AccessibilityMenuListPopup> m_popup;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityMenuList) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isMenuList(); } \
SPECIALIZE_TYPE_TRAITS_END()
