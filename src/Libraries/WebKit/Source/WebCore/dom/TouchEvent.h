/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

#if ENABLE(IOS_TOUCH_EVENTS)
#include <WebKitAdditions/TouchEventIOS.h>
#elif ENABLE(TOUCH_EVENTS)

#include "MouseRelatedEvent.h"
#include "TouchList.h"

namespace WebCore {

class TouchEvent final : public MouseRelatedEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TouchEvent);
public:
    virtual ~TouchEvent();

    static Ref<TouchEvent> create(TouchList* touches, TouchList* targetTouches, TouchList* changedTouches,
        const AtomString& type, RefPtr<WindowProxy>&& view, const IntPoint& globalLocation, OptionSet<Modifier> modifiers)
    {
        return adoptRef(*new TouchEvent(touches, targetTouches, changedTouches, type, WTFMove(view), globalLocation, modifiers));
    }
    static Ref<TouchEvent> createForBindings()
    {
        return adoptRef(*new TouchEvent);
    }

    struct Init : MouseRelatedEventInit {
        RefPtr<TouchList> touches;
        RefPtr<TouchList> targetTouches;
        RefPtr<TouchList> changedTouches;
    };

    static Ref<TouchEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new TouchEvent(type, initializer, isTrusted));
    }

    TouchList* touches() const { return m_touches.get(); }
    TouchList* targetTouches() const { return m_targetTouches.get(); }
    TouchList* changedTouches() const { return m_changedTouches.get(); }

    void setTouches(RefPtr<TouchList>&& touches) { m_touches = touches; }
    void setTargetTouches(RefPtr<TouchList>&& targetTouches) { m_targetTouches = targetTouches; }
    void setChangedTouches(RefPtr<TouchList>&& changedTouches) { m_changedTouches = changedTouches; }

    bool isTouchEvent() const override;

private:
    TouchEvent();
    TouchEvent(TouchList* touches, TouchList* targetTouches, TouchList* changedTouches, const AtomString& type,
        RefPtr<WindowProxy>&&, const IntPoint& globalLocation, OptionSet<Modifier>);
    TouchEvent(const AtomString&, const Init&, IsTrusted);

    RefPtr<TouchList> m_touches;
    RefPtr<TouchList> m_targetTouches;
    RefPtr<TouchList> m_changedTouches;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(TouchEvent)

#endif // ENABLE(TOUCH_EVENTS)
