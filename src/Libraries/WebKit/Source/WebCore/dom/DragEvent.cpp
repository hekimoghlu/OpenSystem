/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#include "DragEvent.h"

#include "DataTransfer.h"
#include "Node.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DragEvent);

Ref<DragEvent> DragEvent::create(const AtomString& eventType, DragEventInit&& init)
{
    return adoptRef(*new DragEvent(eventType, WTFMove(init)));
}

Ref<DragEvent> DragEvent::createForBindings()
{
    return adoptRef(*new DragEvent);
}

Ref<DragEvent> DragEvent::create(const AtomString& type, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&& view, int detail,
    const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier> modifiers, MouseButton button, unsigned short buttons,
    EventTarget* relatedTarget, double force, SyntheticClickType syntheticClickType, DataTransfer* dataTransfer, IsSimulated isSimulated, IsTrusted isTrusted)
{
    return adoptRef(*new DragEvent(type, canBubble, isCancelable, isComposed, timestamp, WTFMove(view), detail,
        screenLocation, windowLocation, movementX, movementY, modifiers, button, buttons, relatedTarget, force, syntheticClickType, dataTransfer, isSimulated, isTrusted));
}

DragEvent::DragEvent(const AtomString& eventType, DragEventInit&& init)
    : MouseEvent(EventInterfaceType::DragEvent, eventType, init, IsTrusted::No)
    , m_dataTransfer(WTFMove(init.dataTransfer))
{
}

DragEvent::DragEvent(const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed,
    MonotonicTime timestamp, RefPtr<WindowProxy>&& view, int detail,
    const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier> modifiers, MouseButton button, unsigned short buttons,
    EventTarget* relatedTarget, double force, SyntheticClickType syntheticClickType, DataTransfer* dataTransfer, IsSimulated isSimulated, IsTrusted isTrusted)
    : MouseEvent(EventInterfaceType::DragEvent, eventType, canBubble, isCancelable, isComposed, timestamp, WTFMove(view), detail, screenLocation, windowLocation, movementX, movementY, modifiers, button, buttons, relatedTarget, force, syntheticClickType, { }, { }, isSimulated, isTrusted)
    , m_dataTransfer(dataTransfer)
{
}

DragEvent::DragEvent()
    : MouseEvent(EventInterfaceType::DragEvent)
{
}

DragEvent::~DragEvent() = default;

} // namespace WebCore
