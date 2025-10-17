/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#include "InputEvent.h"

#include "DataTransfer.h"
#include "Node.h"
#include "WindowProxy.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(InputEvent);

Ref<InputEvent> InputEvent::create(const AtomString& eventType, const String& inputType, IsCancelable cancelable, RefPtr<WindowProxy>&& view,
    const String& data, RefPtr<DataTransfer>&& dataTransfer, const Vector<RefPtr<StaticRange>>& targetRanges, int detail, IsInputMethodComposing isInputMethodComposing)
{
    return adoptRef(*new InputEvent(eventType, inputType, cancelable, WTFMove(view), data, WTFMove(dataTransfer), targetRanges, detail, isInputMethodComposing));
}

InputEvent::InputEvent(const AtomString& eventType, const String& inputType, IsCancelable cancelable, RefPtr<WindowProxy>&& view,
    const String& data, RefPtr<DataTransfer>&& dataTransfer, const Vector<RefPtr<StaticRange>>& targetRanges, int detail, IsInputMethodComposing isInputMethodComposing)
    : UIEvent(EventInterfaceType::InputEvent, eventType, CanBubble::Yes, cancelable, IsComposed::Yes, WTFMove(view), detail)
    , m_inputType(inputType)
    , m_data(data)
    , m_dataTransfer(dataTransfer)
    , m_targetRanges(targetRanges)
    , m_isInputMethodComposing(isInputMethodComposing == IsInputMethodComposing::Yes)
{
}

InputEvent::InputEvent(const AtomString& eventType, const Init& initializer)
    : UIEvent(EventInterfaceType::InputEvent, eventType, initializer)
    , m_inputType(initializer.inputType)
    , m_data(initializer.data)
    , m_isInputMethodComposing(initializer.isComposing)
{
}

InputEvent::~InputEvent() = default;

RefPtr<DataTransfer> InputEvent::dataTransfer() const
{
    return m_dataTransfer;
}

} // namespace WebCore
