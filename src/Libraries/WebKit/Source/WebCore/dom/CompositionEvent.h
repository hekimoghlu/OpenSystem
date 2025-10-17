/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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

#include "UIEvent.h"

namespace WebCore {

class CompositionEvent final : public UIEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CompositionEvent);
public:
    static Ref<CompositionEvent> create(const AtomString& type, RefPtr<WindowProxy>&& view, const String& data)
    {
        return adoptRef(*new CompositionEvent(type, WTFMove(view), data));
    }

    static Ref<CompositionEvent> createForBindings()
    {
        return adoptRef(*new CompositionEvent);
    }

    struct Init : UIEventInit {
        String data;
    };

    static Ref<CompositionEvent> create(const AtomString& type, const Init& initializer)
    {
        return adoptRef(*new CompositionEvent(type, initializer));
    }

    virtual ~CompositionEvent();

    void initCompositionEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&&, const String& data);

    String data() const { return m_data; }

private:
    CompositionEvent();
    CompositionEvent(const AtomString& type, RefPtr<WindowProxy>&&, const String&);
    CompositionEvent(const AtomString& type, const Init&);

    bool isCompositionEvent() const override;

    String m_data;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(CompositionEvent)
