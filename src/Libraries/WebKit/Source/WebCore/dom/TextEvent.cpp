/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#include "TextEvent.h"

#include "DocumentFragment.h"
#include "Editor.h"
#include "EventNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TextEvent);

Ref<TextEvent> TextEvent::createForBindings()
{
    return adoptRef(*new TextEvent);
}

Ref<TextEvent> TextEvent::create(RefPtr<WindowProxy>&& view, const String& data, TextEventInputType inputType)
{
    return adoptRef(*new TextEvent(WTFMove(view), data, inputType));
}

Ref<TextEvent> TextEvent::createForPlainTextPaste(RefPtr<WindowProxy>&& view, const String& data, bool shouldSmartReplace)
{
    return adoptRef(*new TextEvent(WTFMove(view), data, nullptr, TextEventInputPaste, shouldSmartReplace, false, MailBlockquoteHandling::RespectBlockquote));
}

Ref<TextEvent> TextEvent::createForFragmentPaste(RefPtr<WindowProxy>&& view, RefPtr<DocumentFragment>&& data, TextEventInputType inputType, bool shouldSmartReplace, bool shouldMatchStyle, MailBlockquoteHandling mailBlockquoteHandling)
{
    return adoptRef(*new TextEvent(WTFMove(view), emptyString(), WTFMove(data), inputType, shouldSmartReplace, shouldMatchStyle, mailBlockquoteHandling));
}

Ref<TextEvent> TextEvent::createForDrop(RefPtr<WindowProxy>&& view, const String& data)
{
    return adoptRef(*new TextEvent(WTFMove(view), data, TextEventInputDrop));
}

Ref<TextEvent> TextEvent::createForDictation(RefPtr<WindowProxy>&& view, const String& data, const Vector<DictationAlternative>& dictationAlternatives)
{
    return adoptRef(*new TextEvent(WTFMove(view), data, dictationAlternatives));
}

TextEvent::TextEvent()
    : UIEvent(EventInterfaceType::TextEvent)
    , m_shouldSmartReplace(false)
    , m_shouldMatchStyle(false)
    , m_mailBlockquoteHandling(MailBlockquoteHandling::RespectBlockquote)
{
}

TextEvent::TextEvent(RefPtr<WindowProxy>&& view, const String& data, TextEventInputType inputType)
    : UIEvent(EventInterfaceType::TextEvent, eventNames().textInputEvent, CanBubble::Yes, IsCancelable::Yes, IsComposed::Yes, WTFMove(view), 0)
    , m_inputType(inputType)
    , m_data(data)
    , m_shouldSmartReplace(false)
    , m_shouldMatchStyle(false)
    , m_mailBlockquoteHandling(MailBlockquoteHandling::RespectBlockquote)
{
}

TextEvent::TextEvent(RefPtr<WindowProxy>&& view, const String& data, RefPtr<DocumentFragment>&& pastingFragment, TextEventInputType inputType, bool shouldSmartReplace, bool shouldMatchStyle, MailBlockquoteHandling mailBlockquoteHandling)
    : UIEvent(EventInterfaceType::TextEvent, eventNames().textInputEvent, CanBubble::Yes, IsCancelable::Yes, IsComposed::Yes, WTFMove(view), 0)
    , m_inputType(inputType)
    , m_data(data)
    , m_pastingFragment(WTFMove(pastingFragment))
    , m_shouldSmartReplace(shouldSmartReplace)
    , m_shouldMatchStyle(shouldMatchStyle)
    , m_mailBlockquoteHandling(mailBlockquoteHandling)
{
}

TextEvent::TextEvent(RefPtr<WindowProxy>&& view, const String& data, const Vector<DictationAlternative>& dictationAlternatives)
    : UIEvent(EventInterfaceType::TextEvent, eventNames().textInputEvent, CanBubble::Yes, IsCancelable::Yes, IsComposed::Yes, WTFMove(view), 0)
    , m_inputType(TextEventInputDictation)
    , m_data(data)
    , m_shouldSmartReplace(false)
    , m_shouldMatchStyle(false)
    , m_mailBlockquoteHandling(MailBlockquoteHandling::RespectBlockquote)
    , m_dictationAlternatives(dictationAlternatives)
{
}

TextEvent::~TextEvent() = default;

void TextEvent::initTextEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&& view, const String& data)
{
    if (isBeingDispatched())
        return;

    initUIEvent(type, canBubble, cancelable, WTFMove(view), 0);

    m_inputType = TextEventInputKeyboard;

    m_data = data;

    m_pastingFragment = nullptr;
    m_shouldSmartReplace = false;
    m_shouldMatchStyle = false;
    m_mailBlockquoteHandling = MailBlockquoteHandling::RespectBlockquote;
    m_dictationAlternatives = { };
}

bool TextEvent::isTextEvent() const
{
    return true;
}

} // namespace WebCore
