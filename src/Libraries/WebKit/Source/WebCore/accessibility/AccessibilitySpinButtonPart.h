/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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

class AccessibilitySpinButtonPart final : public AccessibilityMockObject {
public:
    static Ref<AccessibilitySpinButtonPart> create(AXID);
    virtual ~AccessibilitySpinButtonPart() = default;

    bool isIncrementor() const final { return m_isIncrementor; }
    void setIsIncrementor(bool value) { m_isIncrementor = value; }

private:
    explicit AccessibilitySpinButtonPart(AXID);

    bool press() final;
    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::SpinButtonPart; }
    bool isSpinButtonPart() const final { return true; }
    LayoutRect elementRect() const final;

    bool m_isIncrementor { true };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilitySpinButtonPart) \
    static bool isType(const WebCore::AXCoreObject& object) \
    { \
        auto* axObject = dynamicDowncast<WebCore::AccessibilityObject>(object); \
        return axObject && axObject->isSpinButtonPart(); \
    } \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isSpinButtonPart(); } \
SPECIALIZE_TYPE_TRAITS_END()
