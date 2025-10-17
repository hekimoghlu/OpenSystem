/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

#if ENABLE(USER_MESSAGE_HANDLERS)

#include "FrameDestructionObserver.h"
#include "UserContentProvider.h"
#include "UserMessageHandler.h"
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class DOMWrapperWorld;
class LocalFrame;
class UserMessageHandler;

class UserMessageHandlersNamespace : public RefCounted<UserMessageHandlersNamespace>, public FrameDestructionObserver, public UserContentProviderInvalidationClient {
public:
    static Ref<UserMessageHandlersNamespace> create(LocalFrame& frame, UserContentProvider& userContentProvider)
    {
        return adoptRef(*new UserMessageHandlersNamespace(frame, userContentProvider));
    }

    virtual ~UserMessageHandlersNamespace();

    Vector<AtomString> supportedPropertyNames() const;
    UserMessageHandler* namedItem(DOMWrapperWorld&, const AtomString&);
    bool isSupportedPropertyName(const AtomString&);

private:
    explicit UserMessageHandlersNamespace(LocalFrame&, UserContentProvider&);

    // UserContentProviderInvalidationClient
    void didInvalidate(UserContentProvider&) override;

    Ref<UserContentProvider> m_userContentProvider;
    UncheckedKeyHashMap<std::pair<AtomString, RefPtr<DOMWrapperWorld>>, RefPtr<UserMessageHandler>> m_messageHandlers;
};

} // namespace WebCore

#endif // ENABLE(USER_MESSAGE_HANDLERS)
