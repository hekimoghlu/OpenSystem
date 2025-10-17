/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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

#include "ExceptionOr.h"
#include "FrameDestructionObserver.h"
#include "UserMessageHandlerDescriptor.h"

namespace WebCore {

class DeferredPromise;

class UserMessageHandler : public RefCounted<UserMessageHandler>, public FrameDestructionObserver {
public:
    static Ref<UserMessageHandler> create(LocalFrame& frame, UserMessageHandlerDescriptor& descriptor)
    {
        return adoptRef(*new UserMessageHandler(frame, descriptor));
    }
    virtual ~UserMessageHandler();

    ExceptionOr<void> postMessage(RefPtr<SerializedScriptValue>&&, Ref<DeferredPromise>&&);

    UserMessageHandlerDescriptor* descriptor() { return m_descriptor.get(); }
    void invalidateDescriptor() { m_descriptor = nullptr; }

private:
    UserMessageHandler(LocalFrame&, UserMessageHandlerDescriptor&);
    
    RefPtr<UserMessageHandlerDescriptor> m_descriptor;
};

} // namespace WebCore

#endif // ENABLE(USER_MESSAGE_HANDLERS)
