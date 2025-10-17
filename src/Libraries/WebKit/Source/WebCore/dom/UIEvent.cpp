/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#include "UIEvent.h"

#include "Node.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
    
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(UIEvent);

UIEvent::UIEvent(enum EventInterfaceType eventInterface)
    : Event(eventInterface)
    , m_detail(0)
{
}

UIEvent::UIEvent(enum EventInterfaceType eventInterface, const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed, RefPtr<WindowProxy>&& viewArg, int detailArg)
    : Event(eventInterface, eventType, canBubble, isCancelable, isComposed)
    , m_view(WTFMove(viewArg))
    , m_detail(detailArg)
{
}

UIEvent::UIEvent(enum EventInterfaceType eventInterface, const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&& viewArg, int detailArg, IsTrusted isTrusted)
    : Event(eventInterface, eventType, canBubble, isCancelable, isComposed, timestamp, isTrusted)
    , m_view(WTFMove(viewArg))
    , m_detail(detailArg)
{
}

UIEvent::UIEvent(enum EventInterfaceType eventInterface, const AtomString& eventType, const UIEventInit& initializer, IsTrusted isTrusted)
    : Event(eventInterface, eventType, initializer, isTrusted)
    , m_view(initializer.view.get())
    , m_detail(initializer.detail)
{
}

UIEvent::~UIEvent() = default;

void UIEvent::initUIEvent(const AtomString& typeArg, bool canBubbleArg, bool cancelableArg, RefPtr<WindowProxy>&& viewArg, int detailArg)
{
    if (isBeingDispatched())
        return;

    initEvent(typeArg, canBubbleArg, cancelableArg);

    m_view = viewArg;
    m_detail = detailArg;
}

bool UIEvent::isUIEvent() const
{
    return true;
}

int UIEvent::layerX()
{
    return 0;
}

int UIEvent::layerY()
{
    return 0;
}

int UIEvent::pageX() const
{
    return 0;
}

int UIEvent::pageY() const
{
    return 0;
}

unsigned UIEvent::which() const
{
    return 0;
}

} // namespace WebCore
