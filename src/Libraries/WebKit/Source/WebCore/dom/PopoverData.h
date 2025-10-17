/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

#include "Element.h"
#include "HTMLElement.h"
#include "HTMLFormControlElement.h"
#include "ToggleEventTask.h"

namespace WebCore {

enum class PopoverVisibilityState : bool {
    Hidden,
    Showing,
};

class PopoverData {
    WTF_MAKE_TZONE_ALLOCATED(PopoverData);
public:
    PopoverData() = default;

    PopoverState popoverState() const { return m_popoverState; }
    void setPopoverState(PopoverState state) { m_popoverState = state; }

    PopoverVisibilityState visibilityState() const { return m_visibilityState; }
    void setVisibilityState(PopoverVisibilityState visibilityState) { m_visibilityState = visibilityState; };

    Element* previouslyFocusedElement() const { return m_previouslyFocusedElement.get(); }
    void setPreviouslyFocusedElement(Element* element) { m_previouslyFocusedElement = element; }

    Ref<ToggleEventTask> ensureToggleEventTask(Element&);

    HTMLElement* invoker() const { return m_invoker.get(); }
    void setInvoker(const HTMLElement* element) { m_invoker = element; }

    class ScopedStartShowingOrHiding {
    public:
    explicit ScopedStartShowingOrHiding(Element& popover)
        : m_popover(popover)
        , m_wasSet(popover.popoverData()->m_isHidingOrShowingPopover)
    {
        m_popover->popoverData()->m_isHidingOrShowingPopover = true;
    }
    ~ScopedStartShowingOrHiding()
    {
        if (!m_wasSet && m_popover->popoverData())
            m_popover->popoverData()->m_isHidingOrShowingPopover = false;
    }
    bool wasShowingOrHiding() const { return m_wasSet; }

    private:
        const Ref<Element> m_popover;
        bool m_wasSet;
    };

private:
    PopoverState m_popoverState;
    PopoverVisibilityState m_visibilityState;
    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_previouslyFocusedElement;
    RefPtr<ToggleEventTask> m_toggleEventTask;
    WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData> m_invoker;
    bool m_isHidingOrShowingPopover = false;
};

}
