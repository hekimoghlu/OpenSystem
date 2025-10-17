/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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
#include "CompositionEvent.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CompositionEvent);

CompositionEvent::CompositionEvent()
    : UIEvent(EventInterfaceType::CompositionEvent)
{
}

CompositionEvent::CompositionEvent(const AtomString& type, RefPtr<WindowProxy>&& view, const String& data)
    : UIEvent(EventInterfaceType::CompositionEvent, type, CanBubble::Yes, IsCancelable::Yes, IsComposed::Yes, WTFMove(view), 0)
    , m_data(data)
{
}

CompositionEvent::CompositionEvent(const AtomString& type, const Init& initializer)
    : UIEvent(EventInterfaceType::CompositionEvent, type, initializer)
    , m_data(initializer.data)
{
}

CompositionEvent::~CompositionEvent() = default;

void CompositionEvent::initCompositionEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&& view, const String& data)
{
    if (isBeingDispatched())
        return;

    initUIEvent(type, canBubble, cancelable, WTFMove(view), 0);

    m_data = data;
}

bool CompositionEvent::isCompositionEvent() const
{
    return true;
}

} // namespace WebCore
