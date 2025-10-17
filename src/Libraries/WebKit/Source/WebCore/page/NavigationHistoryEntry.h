/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#include "EventHandler.h"
#include "EventTarget.h"
#include "HistoryItem.h"
#include "ReferrerPolicy.h"
#include <wtf/RefCounted.h>

namespace JSC {
class JSValue;
}

namespace WebCore {

class Navigation;
class SerializedScriptValue;

class NavigationHistoryEntry final : public RefCounted<NavigationHistoryEntry>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(NavigationHistoryEntry);
public:
    static Ref<NavigationHistoryEntry> create(Navigation&, Ref<HistoryItem>&&);
    static Ref<NavigationHistoryEntry> create(Navigation&, const NavigationHistoryEntry&);

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    const String& url() const;
    String key() const;
    String id() const;
    uint64_t index() const;
    bool sameDocument() const;
    JSC::JSValue getState(JSDOMGlobalObject&) const;

    void setState(RefPtr<SerializedScriptValue>&&);
    SerializedScriptValue* state() const { return m_state.get(); }

    HistoryItem& associatedHistoryItem() const { return m_associatedHistoryItem; }
    void dispatchDisposeEvent();

private:
    struct DocumentState {
        static DocumentState fromContext(ScriptExecutionContext*);

        std::optional<ScriptExecutionContextIdentifier> identifier;
        ReferrerPolicy referrerPolicy { ReferrerPolicy::Default };
    };

    NavigationHistoryEntry(Navigation&, const DocumentState&, Ref<HistoryItem>&&, String urlString, WTF::UUID key, RefPtr<SerializedScriptValue>&& state = { }, WTF::UUID = WTF::UUID::createVersion4());

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final;
    ScriptExecutionContext* scriptExecutionContext() const final;
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    void eventListenersDidChange() final;

    WeakPtr<Navigation, WeakPtrImplWithEventTargetData> m_navigation;
    const String m_urlString;
    const WTF::UUID m_key;
    const WTF::UUID m_id;
    RefPtr<SerializedScriptValue> m_state;
    Ref<HistoryItem> m_associatedHistoryItem;
    DocumentState m_originalDocumentState;
    bool m_hasDisposeEventListener { false };
    bool m_hasDispatchedDisposeEvent { false };
};

} // namespace WebCore
