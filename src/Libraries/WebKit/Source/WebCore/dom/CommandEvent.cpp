/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#include "CommandEvent.h"

#include "Element.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CommandEvent);

CommandEvent::CommandEvent()
    : Event(EventInterfaceType::CommandEvent)
{
}

CommandEvent::CommandEvent(const AtomString& type, const CommandEvent::Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::CommandEvent, type, initializer, isTrusted)
    , m_invoker(initializer.invoker)
    , m_command(initializer.command)
{
}

Ref<CommandEvent> CommandEvent::create(const AtomString& eventType, const CommandEvent::Init& init, IsTrusted isTrusted)
{
    return adoptRef(*new CommandEvent(eventType, init, isTrusted));
}

Ref<CommandEvent> CommandEvent::createForBindings()
{
    return adoptRef(*new CommandEvent);
}

bool CommandEvent::isCommandEvent() const
{
    return true;
}

RefPtr<Element> CommandEvent::invoker() const
{
    auto* invoker = m_invoker.get();
    if (!invoker)
        return nullptr;

    if (RefPtr target = dynamicDowncast<Node>(currentTarget())) {
        auto& treeScope = target->treeScope();
        auto node = treeScope.retargetToScope(*invoker);
        return &downcast<Element>(node).get();
    }
    return invoker;
}

} // namespace WebCore
