/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#include "config.h"
#include "AccessibilitySpinButtonPart.h"

#include "AccessibilitySpinButton.h"

namespace WebCore {

AccessibilitySpinButtonPart::AccessibilitySpinButtonPart(AXID axID)
    : AccessibilityMockObject(axID)
    , m_isIncrementor(false)
{
}

Ref<AccessibilitySpinButtonPart> AccessibilitySpinButtonPart::create(AXID axID)
{
    return adoptRef(*new AccessibilitySpinButtonPart(axID));
}

LayoutRect AccessibilitySpinButtonPart::elementRect() const
{
    // FIXME: This logic should exist in the render tree or elsewhere, but there is no
    // relationship that exists that can be queried.

    RefPtr parent = parentObject();
    if (!parent)
        return { };

    LayoutRect parentRect = parent->elementRect();
    if (m_isIncrementor)
        parentRect.setHeight(parentRect.height() / 2);
    else {
        parentRect.setY(parentRect.y() + parentRect.height() / 2);
        parentRect.setHeight(parentRect.height() / 2);
    }

    return parentRect;
}

bool AccessibilitySpinButtonPart::press()
{
    RefPtr spinButton = dynamicDowncast<AccessibilitySpinButton>(m_parent.get());
    if (!spinButton)
        return false;

    if (m_isIncrementor)
        spinButton->step(1);
    else
        spinButton->step(-1);

    return true;
}

} // namespace WebCore
