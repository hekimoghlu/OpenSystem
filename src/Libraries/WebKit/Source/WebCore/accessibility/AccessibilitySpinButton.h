/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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
#include "AccessibilitySpinButtonPart.h"
#include "SpinButtonElement.h"

namespace WebCore {

// Currently only represents native spinbuttons (i.e. <input type="number">) and not role="spinbutton" elements.
class AccessibilitySpinButton final : public AccessibilityMockObject {
public:
    static Ref<AccessibilitySpinButton> create(AXID, AXObjectCache&);
    virtual ~AccessibilitySpinButton();

    void setSpinButtonElement(SpinButtonElement* spinButton) { m_spinButtonElement = spinButton; }

    AccessibilitySpinButtonPart* incrementButton() final;
    AccessibilitySpinButtonPart* decrementButton() final;

    void step(int amount);

private:
    explicit AccessibilitySpinButton(AXID, AXObjectCache&);

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::SpinButton; }
    bool isNativeSpinButton() const final { return true; }
    void clearChildren() final { };
    void addChildren() final;
    LayoutRect elementRect() const final;

    WeakPtr<SpinButtonElement, WeakPtrImplWithEventTargetData> m_spinButtonElement;
    // FIXME: Nothing calls AXObjectCache::remove for m_incrementor and m_decrementor.
    Ref<AccessibilitySpinButtonPart> m_incrementor;
    Ref<AccessibilitySpinButtonPart> m_decrementor;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilitySpinButton) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isNativeSpinButton(); } \
SPECIALIZE_TYPE_TRAITS_END()
