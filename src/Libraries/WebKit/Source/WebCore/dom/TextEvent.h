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
#pragma once

#include "DictationAlternative.h"
#include "TextEventInputType.h"
#include "UIEvent.h"

namespace WebCore {

    class DocumentFragment;

    enum class MailBlockquoteHandling : bool;

    class TextEvent final : public UIEvent {
        WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextEvent);
    public:
        static Ref<TextEvent> create(RefPtr<WindowProxy>&&, const String& data, TextEventInputType = TextEventInputKeyboard);
        static Ref<TextEvent> createForBindings();
        static Ref<TextEvent> createForPlainTextPaste(RefPtr<WindowProxy>&&, const String& data, bool shouldSmartReplace);
        static Ref<TextEvent> createForFragmentPaste(RefPtr<WindowProxy>&&, RefPtr<DocumentFragment>&& data, TextEventInputType, bool shouldSmartReplace, bool shouldMatchStyle, MailBlockquoteHandling);
        static Ref<TextEvent> createForDrop(RefPtr<WindowProxy>&&, const String& data);
        static Ref<TextEvent> createForDictation(RefPtr<WindowProxy>&&, const String& data, const Vector<DictationAlternative>& dictationAlternatives);

        virtual ~TextEvent();
    
        WEBCORE_EXPORT void initTextEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&&, const String& data);
    
        String data() const { return m_data; }

        bool isLineBreak() const { return m_inputType == TextEventInputLineBreak; }
        bool isComposition() const { return m_inputType == TextEventInputComposition; }
        bool isBackTab() const { return m_inputType == TextEventInputBackTab; }
        bool isPaste() const { return m_inputType == TextEventInputPaste; }
        bool isDrop() const { return m_inputType == TextEventInputDrop; }
        bool isDictation() const { return m_inputType == TextEventInputDictation; }
        bool isAutocompletion() const { return m_inputType == TextEventInputAutocompletion; }
        bool isKeyboard() const { return m_inputType == TextEventInputKeyboard; }
        bool isRemoveBackground() const { return m_inputType == TextEventInputRemoveBackground; }

        bool shouldSmartReplace() const { return m_shouldSmartReplace; }
        bool shouldMatchStyle() const { return m_shouldMatchStyle; }
        MailBlockquoteHandling mailBlockquoteHandling() const { return m_mailBlockquoteHandling; }
        DocumentFragment* pastingFragment() const { return m_pastingFragment.get(); }
        const Vector<DictationAlternative>& dictationAlternatives() const { return m_dictationAlternatives; }

    private:
        TextEvent();

        TextEvent(RefPtr<WindowProxy>&&, const String& data, TextEventInputType = TextEventInputKeyboard);
        TextEvent(RefPtr<WindowProxy>&&, const String& data, RefPtr<DocumentFragment>&&, TextEventInputType, bool shouldSmartReplace, bool shouldMatchStyle, MailBlockquoteHandling);
        TextEvent(RefPtr<WindowProxy>&&, const String& data, const Vector<DictationAlternative>& dictationAlternatives);

        bool isTextEvent() const override;

        TextEventInputType m_inputType;
        String m_data;

        RefPtr<DocumentFragment> m_pastingFragment;
        bool m_shouldSmartReplace;
        bool m_shouldMatchStyle;
        MailBlockquoteHandling m_mailBlockquoteHandling;
        Vector<DictationAlternative> m_dictationAlternatives;
    };

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(TextEvent)
