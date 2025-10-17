/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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

#include "StaticRange.h"
#include "UIEvent.h"

namespace WebCore {

class DataTransfer;
class WindowProxy;

enum class IsInputMethodComposing : bool { No, Yes };

class InputEvent final : public UIEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InputEvent);
public:
    struct Init : UIEventInit {
        String data;
        bool isComposing { false }; // input method
        String inputType;
    };

    virtual ~InputEvent();

    static Ref<InputEvent> create(const AtomString& eventType, const String& inputType, IsCancelable, RefPtr<WindowProxy>&& view,
        const String& data, RefPtr<DataTransfer>&&, const Vector<RefPtr<StaticRange>>& targetRanges, int detail, IsInputMethodComposing);

    static Ref<InputEvent> create(const AtomString& type, const Init& initializer)
    {
        return adoptRef(*new InputEvent(type, initializer));
    }

    bool isInputEvent() const override { return true; }
    const String& inputType() const { return m_inputType; }
    const String& data() const { return m_data; }
    RefPtr<DataTransfer> dataTransfer() const;
    const Vector<RefPtr<StaticRange>>& getTargetRanges() { return m_targetRanges; }
    bool isInputMethodComposing() const { return m_isInputMethodComposing; }

private:
    InputEvent(const AtomString& eventType, const String& inputType, IsCancelable, RefPtr<WindowProxy>&&,
        const String& data, RefPtr<DataTransfer>&&, const Vector<RefPtr<StaticRange>>& targetRanges, int detail, IsInputMethodComposing);
    InputEvent(const AtomString& eventType, const Init&);

    String m_inputType;
    String m_data;
    RefPtr<DataTransfer> m_dataTransfer;
    Vector<RefPtr<StaticRange>> m_targetRanges;
    bool m_isInputMethodComposing;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(InputEvent)
