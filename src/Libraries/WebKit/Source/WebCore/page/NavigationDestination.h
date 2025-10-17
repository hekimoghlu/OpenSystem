/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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

#include "EventHandler.h"
#include "LocalDOMWindowProperty.h"
#include "NavigationHistoryEntry.h"
#include "ScriptWrappable.h"

namespace JSC {
class JSValue;
}

namespace WebCore {

class NavigationDestination final : public RefCounted<NavigationDestination>, public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(NavigationDestination);
public:
    static Ref<NavigationDestination> create(const URL& url, RefPtr<NavigationHistoryEntry>&& entry, bool isSameDocument) { return adoptRef(*new NavigationDestination(url, WTFMove(entry), isSameDocument)); };

    const URL& url() const { return m_url; };
    String key() const { return m_entry ? m_entry->key() : String(); };
    String id() const { return m_entry ? m_entry->id() : String(); };
    int64_t index() const { return m_entry ? m_entry->index() : -1; };
    bool sameDocument() const { return m_isSameDocument; };
    JSC::JSValue getState(JSDOMGlobalObject&) const;
    void setStateObject(SerializedScriptValue* stateObject) { m_stateObject = stateObject; }

private:
    explicit NavigationDestination(const URL&, RefPtr<NavigationHistoryEntry>&&, bool isSameDocument);

    RefPtr<NavigationHistoryEntry> m_entry;
    URL m_url;
    bool m_isSameDocument;
    RefPtr<SerializedScriptValue> m_stateObject;
};

} // namespace WebCore
