/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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

#include "AccessibilityNodeObject.h"
#include "AccessibilityObject.h"

namespace WebCore {

class HTMLOptionElement;

class AccessibilityMenuListOption final : public AccessibilityNodeObject {
public:
    static Ref<AccessibilityMenuListOption> create(AXID, HTMLOptionElement&);
    void setParent(AccessibilityObject* parent) { m_parent = parent; }

private:
    explicit AccessibilityMenuListOption(AXID, HTMLOptionElement&);

    bool isMenuListOption() const final { return true; }

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::MenuListOption; }
    bool canHaveChildren() const final { return false; }
    AccessibilityObject* parentObject() const final { return m_parent.get(); }

    HTMLOptionElement* optionElement() const;
    Element* actionElement() const final;
    bool isEnabled() const final;
    bool isVisible() const final;
    bool isOffScreen() const final;
    bool isSelected() const final;
    void setSelected(bool) final;
    bool canSetSelectedAttribute() const final;
    LayoutRect elementRect() const final;
    String stringValue() const final;
    bool computeIsIgnored() const final;

    WeakPtr<AccessibilityObject> m_parent;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityMenuListOption) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isMenuListOption(); } \
SPECIALIZE_TYPE_TRAITS_END()
