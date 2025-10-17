/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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

#include "Event.h"
#include "UIEventInit.h"
#include "WindowProxy.h"

namespace WebCore {

// FIXME: Remove this when no one is depending on it anymore.
typedef WindowProxy AbstractView;

class UIEvent : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(UIEvent);
public:
    static Ref<UIEvent> create(const AtomString& type, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed, RefPtr<WindowProxy>&& view, int detail)
    {
        return adoptRef(*new UIEvent(EventInterfaceType::UIEvent, type, canBubble, isCancelable, isComposed, WTFMove(view), detail));
    }
    static Ref<UIEvent> createForBindings()
    {
        return adoptRef(*new UIEvent(EventInterfaceType::UIEvent));
    }
    static Ref<UIEvent> create(const AtomString& type, const UIEventInit& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new UIEvent(EventInterfaceType::UIEvent, type, initializer, isTrusted));
    }
    virtual ~UIEvent();

    WEBCORE_EXPORT void initUIEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&&, int detail);

    WindowProxy* view() const { return m_view.get(); }
    int detail() const { return m_detail; }

    virtual int layerX();
    virtual int layerY();

    virtual int pageX() const;
    virtual int pageY() const;

    virtual unsigned which() const;

protected:
    UIEvent(enum EventInterfaceType);

    UIEvent(enum EventInterfaceType, const AtomString& type, CanBubble, IsCancelable, IsComposed, RefPtr<WindowProxy>&&, int detail);
    UIEvent(enum EventInterfaceType, const AtomString& type, CanBubble, IsCancelable, IsComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&&, int detail, IsTrusted = IsTrusted::Yes);
    UIEvent(enum EventInterfaceType, const AtomString&, const UIEventInit&, IsTrusted = IsTrusted::No);

private:
    bool isUIEvent() const final;

    RefPtr<WindowProxy> m_view;
    int m_detail;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(UIEvent)
