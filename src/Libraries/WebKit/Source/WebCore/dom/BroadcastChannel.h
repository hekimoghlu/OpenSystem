/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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

#include "ActiveDOMObject.h"
#include "BroadcastChannelIdentifier.h"
#include "ClientOrigin.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>

namespace JSC {
class JSGlobalObject;
class JSValue;
}

namespace WebCore {

class SerializedScriptValue;

class BroadcastChannel : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<BroadcastChannel>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(BroadcastChannel);
public:
    static Ref<BroadcastChannel> create(ScriptExecutionContext& context, const String& name)
    {
        auto channel = adoptRef(*new BroadcastChannel(context, name));
        channel->suspendIfNeeded();
        return channel;
    }
    ~BroadcastChannel();

    // ActiveDOMObject.
    void ref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void deref() const final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }

    BroadcastChannelIdentifier identifier() const;
    String name() const;

    ExceptionOr<void> postMessage(JSC::JSGlobalObject&, JSC::JSValue message);
    void close();

    WEBCORE_EXPORT static void dispatchMessageTo(BroadcastChannelIdentifier, Ref<SerializedScriptValue>&&, CompletionHandler<void()>&&);

private:
    BroadcastChannel(ScriptExecutionContext&, const String& name);

    void dispatchMessage(Ref<SerializedScriptValue>&&);

    bool isEligibleForMessaging() const;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::BroadcastChannel; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    void refEventTarget() final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::ref(); }
    void derefEventTarget() final { ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr::deref(); }
    void eventListenersDidChange() final;

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;
    void stop() final { close(); }

    class MainThreadBridge;
    Ref<MainThreadBridge> protectedMainThreadBridge() const;

    Ref<MainThreadBridge> m_mainThreadBridge;
    bool m_isClosed { false };
    bool m_hasRelevantEventListener { false };
};

} // namespace WebCore
