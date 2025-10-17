/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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

#include "InputType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SliderThumbElement;

class RangeInputType final : public InputType {
    WTF_MAKE_TZONE_ALLOCATED(RangeInputType);
public:
    static Ref<RangeInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new RangeInputType(element));
    }

    bool typeMismatchFor(const String&) const final;

private:
    explicit RangeInputType(HTMLInputElement&);

    const AtomString& formControlType() const final;
    double valueAsDouble() const final;
    ExceptionOr<void> setValueAsDecimal(const Decimal&, TextFieldEventBehavior) const final;
    bool supportsRequired() const final;
    StepRange createStepRange(AnyStepHandling) const final;
    void handleMouseDownEvent(MouseEvent&) final;
    ShouldCallBaseEventHandler handleKeydownEvent(KeyboardEvent&) final;
    RenderPtr<RenderElement> createInputRenderer(RenderStyle&&) final;
    void createShadowSubtree() final;
    Decimal parseToNumber(const String&, const Decimal&) const final;
    String serialize(const Decimal&) const final;
    bool accessKeyAction(bool sendMouseEvents) final;
    void attributeChanged(const QualifiedName&) final;
    void setValue(const String&, bool valueChanged, TextFieldEventBehavior, TextControlSetValueSelection) final;
    String fallbackValue() const final;
    String sanitizeValue(const String& proposedValue) const final;
    bool shouldRespectListAttribute() final;
    HTMLElement* sliderThumbElement() const final;
    HTMLElement* sliderTrackElement() const final;

    SliderThumbElement& typedSliderThumbElement() const;

    void dataListMayHaveChanged() final;
    void updateTickMarkValues();
    std::optional<Decimal> findClosestTickMarkValue(const Decimal&) final;

    bool m_tickMarkValuesDirty { true };
    Vector<Decimal> m_tickMarkValues;

#if ENABLE(TOUCH_EVENTS)
    void handleTouchEvent(TouchEvent&) final;
#endif

    void disabledStateChanged() final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(RangeInputType, Type::Range)
