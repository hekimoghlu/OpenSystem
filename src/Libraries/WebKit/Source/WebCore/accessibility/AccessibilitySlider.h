/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "AccessibilityRenderObject.h"

namespace WebCore {

class HTMLInputElement;

class AccessibilitySlider final : public AccessibilityRenderObject {
public:
    static Ref<AccessibilitySlider> create(AXID, RenderObject&);
    virtual ~AccessibilitySlider() = default;

private:
    explicit AccessibilitySlider(AXID, RenderObject&);

    HTMLInputElement* inputElement() const;
    AccessibilityObject* elementAccessibilityHitTest(const IntPoint&) const final;

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::Slider; }

    void addChildren() final;

    bool canSetValueAttribute() const final { return true; }
    
    bool setValue(const String&) final;
    float valueForRange() const final;
    float maxValueForRange() const final;
    float minValueForRange() const final;
    AccessibilityOrientation orientation() const final;
};

class AccessibilitySliderThumb final : public AccessibilityMockObject {
public:
    static Ref<AccessibilitySliderThumb> create(AXID);
    virtual ~AccessibilitySliderThumb() = default;

    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::SliderThumb; }
    LayoutRect elementRect() const final;

private:
    explicit AccessibilitySliderThumb(AXID);

    bool isSliderThumb() const final { return true; }
    bool computeIsIgnored() const final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilitySliderThumb) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isSliderThumb(); } \
SPECIALIZE_TYPE_TRAITS_END()
