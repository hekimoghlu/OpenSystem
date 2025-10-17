/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#include "UserMessageHandlersNamespace.h"

#if ENABLE(USER_MESSAGE_HANDLERS)

#include "DOMWrapperWorld.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"
#include "Page.h"
#include "UserContentController.h"
#include "UserMessageHandler.h"

namespace WebCore {

UserMessageHandlersNamespace::UserMessageHandlersNamespace(LocalFrame& frame, UserContentProvider& userContentProvider)
    : FrameDestructionObserver(&frame)
    , m_userContentProvider(userContentProvider)
{
    m_userContentProvider->registerForUserMessageHandlerInvalidation(*this);
}

UserMessageHandlersNamespace::~UserMessageHandlersNamespace()
{
    m_userContentProvider->unregisterForUserMessageHandlerInvalidation(*this);
}

void UserMessageHandlersNamespace::didInvalidate(UserContentProvider& provider)
{
    auto oldMap = WTFMove(m_messageHandlers);

    provider.forEachUserMessageHandler([&](const UserMessageHandlerDescriptor& descriptor) {
        auto userMessageHandler = oldMap.take(std::make_pair(descriptor.name(), const_cast<DOMWrapperWorld*>(&descriptor.world())));
        if (userMessageHandler) {
            m_messageHandlers.add(std::make_pair(descriptor.name(), const_cast<DOMWrapperWorld*>(&descriptor.world())), userMessageHandler);
            return;
        }
    });

    for (auto& userMessageHandler : oldMap.values())
        userMessageHandler->invalidateDescriptor();
}

Vector<AtomString> UserMessageHandlersNamespace::supportedPropertyNames() const
{
    // FIXME: Consider adding support for iterating the registered UserMessageHandlers. This would
    // require adding support for passing the DOMWrapperWorld to supportedPropertyNames.
    return { };
}

bool UserMessageHandlersNamespace::isSupportedPropertyName(const AtomString&)
{
    // FIXME: Consider adding support for this. It would require adding support for passing the
    // DOMWrapperWorld to isSupportedPropertyName().
    return false;
}

UserMessageHandler* UserMessageHandlersNamespace::namedItem(DOMWrapperWorld& world, const AtomString& name)
{
    auto* frame = this->frame();
    if (!frame)
        return nullptr;

    Page* page = frame->page();
    if (!page)
        return nullptr;

    UserMessageHandler* handler = m_messageHandlers.get(std::pair<AtomString, RefPtr<DOMWrapperWorld>>(name, &world));
    if (handler)
        return handler;

    page->protectedUserContentProvider()->forEachUserMessageHandler([&](const UserMessageHandlerDescriptor& descriptor) {
        if (descriptor.name() != name || &descriptor.world() != &world)
            return;
        
        ASSERT(!handler);

        auto addResult = m_messageHandlers.add(std::make_pair(descriptor.name(), const_cast<DOMWrapperWorld*>(&descriptor.world())), UserMessageHandler::create(*frame, const_cast<UserMessageHandlerDescriptor&>(descriptor)));
        handler = addResult.iterator->value.get();
    });

    return handler;
}

} // namespace WebCore

#endif // ENABLE(USER_MESSAGE_HANDLERS)
