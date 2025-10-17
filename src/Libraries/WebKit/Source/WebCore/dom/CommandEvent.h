/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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

#include "Event.h"
#include "EventInit.h"
#include <wtf/Forward.h>

namespace WebCore {

class Element;
class HTMLElement;

class CommandEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CommandEvent);

public:
    struct Init : EventInit {
        RefPtr<Element> invoker;
        String command;
    };

    static Ref<CommandEvent> create(const AtomString& type, const Init&, IsTrusted = IsTrusted::No);
    static Ref<CommandEvent> createForBindings();

    RefPtr<Element> invoker() const;

    String command() const { return m_command; }

private:
    CommandEvent();
    CommandEvent(const AtomString& type, const Init&, IsTrusted = IsTrusted::No);

    bool isCommandEvent() const final;

    void setCommandr(RefPtr<Element>&& invoker) { m_invoker = WTFMove(invoker); }

    RefPtr<Element> m_invoker;
    String m_command;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(CommandEvent)
