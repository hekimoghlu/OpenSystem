/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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

#include "EventTarget.h"
#include "UIEvent.h"

namespace WebCore {

class Node;

class FocusEvent final : public UIEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FocusEvent);
public:
    static Ref<FocusEvent> create(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, RefPtr<WindowProxy>&& view, int detail, RefPtr<EventTarget>&& relatedTarget)
    {
        return adoptRef(*new FocusEvent(type, canBubble, cancelable, WTFMove(view), detail, WTFMove(relatedTarget)));
    }

    static Ref<FocusEvent> createForBindings()
    {
        return adoptRef(*new FocusEvent);
    }

    struct Init : UIEventInit {
        RefPtr<EventTarget> relatedTarget;
    };

    static Ref<FocusEvent> create(const AtomString& type, const Init& initializer)
    {
        return adoptRef(*new FocusEvent(type, initializer));
    }

    EventTarget* relatedTarget() const final { return m_relatedTarget.get(); }

private:
    FocusEvent();
    FocusEvent(const AtomString& type, CanBubble, IsCancelable, RefPtr<WindowProxy>&&, int, RefPtr<EventTarget>&&);
    FocusEvent(const AtomString& type, const Init&);

    bool isFocusEvent() const final;

    void setRelatedTarget(RefPtr<EventTarget>&& relatedTarget) final { m_relatedTarget = WTFMove(relatedTarget); }

    RefPtr<EventTarget> m_relatedTarget;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(FocusEvent)
