/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#include "HTMLDivElement.h"
#include "PopupOpeningObserver.h"
#include "Timer.h"
#include <wtf/WeakPtr.h>

namespace WebCore {
class SpinButtonOwner;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SpinButtonOwner> : std::true_type { };
}

namespace WebCore {

class SpinButtonOwner : public CanMakeWeakPtr<SpinButtonOwner> {
public:
    virtual ~SpinButtonOwner() = default;
    virtual void focusAndSelectSpinButtonOwner() = 0;
    virtual bool shouldSpinButtonRespondToMouseEvents() const = 0;
    virtual void spinButtonStepDown() = 0;
    virtual void spinButtonStepUp() = 0;
};

class SpinButtonElement final : public HTMLDivElement, public PopupOpeningObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SpinButtonElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SpinButtonElement);
public:
    enum UpDownState {
        Indeterminate, // Hovered, but the event is not handled.
        Down,
        Up,
    };

    // The owner of SpinButtonElement must call removeSpinButtonOwner
    // because SpinButtonElement can be outlive SpinButtonOwner
    // implementation, e.g. during event handling.
    static Ref<SpinButtonElement> create(Document&, SpinButtonOwner&);
    UpDownState upDownState() const { return m_upDownState; }
    void releaseCapture();
    void removeSpinButtonOwner() { m_spinButtonOwner = nullptr; }

    USING_CAN_MAKE_WEAKPTR(HTMLDivElement);

    void step(int amount);
    
    bool willRespondToMouseMoveEvents() const override;
    bool willRespondToMouseClickEventsWithEditability(Editability) const override;

private:
    SpinButtonElement(Document&, SpinButtonOwner&);

    void willDetachRenderers() override;
    bool isSpinButtonElement() const override { return true; }
    bool isDisabledFormControl() const override { return shadowHost() && shadowHost()->isDisabledFormControl(); }
    bool matchesReadWritePseudoClass() const override;
    void defaultEventHandler(Event&) override;
    void willOpenPopup() override;
    void doStepAction(int);
    void startRepeatingTimer();
    void stopRepeatingTimer();
    void repeatingTimerFired();
    void setHovered(bool, Style::InvalidationScope, HitTestRequest) override;
    bool shouldRespondToMouseEvents() const;
    bool isMouseFocusable() const override { return false; }

    WeakPtr<SpinButtonOwner> m_spinButtonOwner;
    bool m_capturing;
    UpDownState m_upDownState;
    UpDownState m_pressStartingState;
    Timer m_repeatingTimer;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SpinButtonElement)
    static bool isType(const WebCore::Element& element) { return element.isSpinButtonElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* element = dynamicDowncast<WebCore::Element>(node);
        return element && isType(*element);
    }
SPECIALIZE_TYPE_TRAITS_END()
