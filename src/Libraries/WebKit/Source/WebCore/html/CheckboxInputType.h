/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

#include "BaseCheckableInputType.h"
#include "SwitchTrigger.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

#if ENABLE(IOS_TOUCH_EVENTS)
class Touch;
#endif

enum class WasSetByJavaScript : bool;
enum class SwitchAnimationType : bool { VisuallyOn, Held };

class CheckboxInputType final : public BaseCheckableInputType {
    WTF_MAKE_TZONE_ALLOCATED(CheckboxInputType);
public:
    static Ref<CheckboxInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new CheckboxInputType(element));
    }

    bool valueMissing(const String&) const final;
    float switchAnimationVisuallyOnProgress() const;
    bool isSwitchVisuallyOn() const;
    float switchAnimationHeldProgress() const;
    bool isSwitchHeld() const;

private:
    explicit CheckboxInputType(HTMLInputElement& element)
        : BaseCheckableInputType(Type::Checkbox, element)
    {
    }

    const AtomString& formControlType() const final;
    String valueMissingText() const final;
    void createShadowSubtree() final;
    void handleKeyupEvent(KeyboardEvent&) final;
    void handleMouseDownEvent(MouseEvent&) final;
    void handleMouseMoveEvent(MouseEvent&) final;
// FIXME: It should not be iOS-specific, but it's not been tested with a non-iOS touch
// implementation thus far.
#if ENABLE(IOS_TOUCH_EVENTS)
    Touch* subsequentTouchEventTouch(const TouchEvent&) const;
    void handleTouchEvent(TouchEvent&) final;
#endif
    void startSwitchPointerTracking(LayoutPoint);
    void stopSwitchPointerTracking();
    bool isSwitchPointerTracking() const;
    void willDispatchClick(InputElementClickState&) final;
    void didDispatchClick(Event&, const InputElementClickState&) final;
    bool matchesIndeterminatePseudoClass() const final;
    void willUpdateCheckedness(bool /* nowChecked */, WasSetByJavaScript);
    void disabledStateChanged() final;
    Seconds switchAnimationStartTime(SwitchAnimationType) const;
    void setSwitchAnimationStartTime(SwitchAnimationType, Seconds);
    bool isSwitchAnimating(SwitchAnimationType) const;
    void performSwitchAnimation(SwitchAnimationType);
    void performSwitchVisuallyOnAnimation(SwitchTrigger);
    void setIsSwitchHeld(bool /* isHeld */);
    void stopSwitchAnimation(SwitchAnimationType);
    float switchAnimationProgress(SwitchAnimationType) const;
    void updateIsSwitchVisuallyOnFromAbsoluteLocation(LayoutPoint);
    void switchAnimationTimerFired();

    // FIXME: Consider moving all switch-related state (and methods?) to their own object so
    // CheckboxInputType can stay somewhat small.
    std::optional<int> m_switchPointerTrackingLogicalLeftPositionStart { std::nullopt };
    bool m_hasSwitchVisuallyOnChanged { false };
    bool m_isSwitchVisuallyOn { false };
    bool m_isSwitchHeld { false };
    Seconds m_switchAnimationVisuallyOnStartTime { 0_s };
    Seconds m_switchAnimationHeldStartTime { 0_s };
    std::unique_ptr<Timer> m_switchAnimationTimer;
#if ENABLE(IOS_TOUCH_EVENTS)
    std::unique_ptr<Timer> m_switchHeldTimer;
    std::optional<unsigned> m_switchPointerTrackingTouchIdentifier { std::nullopt };
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(CheckboxInputType, Type::Checkbox)
