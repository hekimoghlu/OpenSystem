/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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
#include "FocusEvent.h"

#include "Node.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FocusEvent);

bool FocusEvent::isFocusEvent() const
{
    return true;
}

FocusEvent::FocusEvent()
    : UIEvent(EventInterfaceType::FocusEvent)
{
}

FocusEvent::FocusEvent(const AtomString& type, CanBubble canBubble, IsCancelable isCancelable, RefPtr<WindowProxy>&& view, int detail, RefPtr<EventTarget>&& relatedTarget)
    : UIEvent(EventInterfaceType::FocusEvent, type, canBubble, isCancelable, IsComposed::Yes, WTFMove(view), detail)
    , m_relatedTarget(WTFMove(relatedTarget))
{
}

FocusEvent::FocusEvent(const AtomString& type, const Init& initializer)
    : UIEvent(EventInterfaceType::FocusEvent, type, initializer)
    , m_relatedTarget(initializer.relatedTarget)
{
}

} // namespace WebCore
