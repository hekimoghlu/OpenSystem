/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#include "NavigationHistoryEntry.h"

#include "EventNames.h"
#include "FrameLoader.h"
#include "HistoryController.h"
#include "JSDOMGlobalObject.h"
#include "LocalDOMWindow.h"
#include "Navigation.h"
#include "ScriptExecutionContext.h"
#include "SerializedScriptValue.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NavigationHistoryEntry);

NavigationHistoryEntry::NavigationHistoryEntry(Navigation& navigation, const DocumentState& originalDocumentState, Ref<HistoryItem>&& historyItem, String urlString, WTF::UUID key, RefPtr<SerializedScriptValue>&& state, WTF::UUID id)
    : ActiveDOMObject(navigation.protectedScriptExecutionContext().get())
    , m_navigation(navigation)
    , m_urlString(urlString)
    , m_key(key)
    , m_id(id)
    , m_state(state)
    , m_associatedHistoryItem(WTFMove(historyItem))
    , m_originalDocumentState(originalDocumentState)
{
}

Ref<NavigationHistoryEntry> NavigationHistoryEntry::create(Navigation& navigation, Ref<HistoryItem>&& historyItem)
{
    Ref entry = adoptRef(*new NavigationHistoryEntry(navigation, DocumentState::fromContext(navigation.protectedScriptExecutionContext().get()), WTFMove(historyItem), historyItem->urlString(), historyItem->uuidIdentifier()));
    entry->suspendIfNeeded();
    return entry;
}

Ref<NavigationHistoryEntry> NavigationHistoryEntry::create(Navigation& navigation, const NavigationHistoryEntry& other)
{
    Ref historyItem = other.m_associatedHistoryItem;
    RefPtr state = historyItem->navigationAPIStateObject();
    if (!state)
        state = other.m_state;
    Ref entry = adoptRef(*new NavigationHistoryEntry(navigation, DocumentState::fromContext(other.scriptExecutionContext()), WTFMove(historyItem), other.m_urlString, other.m_key, WTFMove(state), other.m_id));
    entry->suspendIfNeeded();
    return entry;
}

ScriptExecutionContext* NavigationHistoryEntry::scriptExecutionContext() const
{
    return ContextDestructionObserver::scriptExecutionContext();
}

enum EventTargetInterfaceType NavigationHistoryEntry::eventTargetInterface() const
{
    return EventTargetInterfaceType::NavigationHistoryEntry;
}

void NavigationHistoryEntry::eventListenersDidChange()
{
    m_hasDisposeEventListener = hasEventListeners(eventNames().disposeEvent);
}

const String& NavigationHistoryEntry::url() const
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document || !document->isFullyActive())
        return nullString();
    // https://html.spec.whatwg.org/#dom-navigationhistoryentry-url (Step 4)
    if (document->identifier() != m_originalDocumentState.identifier && (m_originalDocumentState.referrerPolicy == ReferrerPolicy::NoReferrer || m_originalDocumentState.referrerPolicy == ReferrerPolicy::Origin))
        return nullString();
    return m_urlString;
}

String NavigationHistoryEntry::key() const
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document || !document->isFullyActive())
        return nullString();
    return m_key.toString();
}

String NavigationHistoryEntry::id() const
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document || !document->isFullyActive())
        return nullString();
    return m_id.toString();
}

uint64_t NavigationHistoryEntry::index() const
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document || !document->isFullyActive())
        return -1;
    return document->domWindow()->navigation().entries().findIf([this] (auto& entry) {
        return entry.ptr() == this;
    });
}

// https://html.spec.whatwg.org/multipage/nav-history-apis.html#dom-navigationhistoryentry-samedocument
bool NavigationHistoryEntry::sameDocument() const
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document || !document->isFullyActive())
        return false;
    RefPtr currentItem = document->frame() ? document->frame()->loader().history().currentItem() : nullptr;
    if (!currentItem)
        return false;
    return currentItem->documentSequenceNumber() == m_associatedHistoryItem->documentSequenceNumber();
}

JSC::JSValue NavigationHistoryEntry::getState(JSDOMGlobalObject& globalObject) const
{
    RefPtr document = dynamicDowncast<Document>(scriptExecutionContext());
    if (!document || !document->isFullyActive())
        return JSC::jsUndefined();

    if (!m_state)
        return JSC::jsUndefined();

    return m_state->deserialize(globalObject, &globalObject, SerializationErrorMode::Throwing);
}

void NavigationHistoryEntry::setState(RefPtr<SerializedScriptValue>&& state)
{
    m_state = state;
    m_associatedHistoryItem->setNavigationAPIStateObject(WTFMove(state));
}

auto NavigationHistoryEntry::DocumentState::fromContext(ScriptExecutionContext* context) -> DocumentState
{
    if (!context)
        return { };
    return { context->identifier(), context->referrerPolicy() };
}

bool NavigationHistoryEntry::virtualHasPendingActivity() const
{
    return m_hasDisposeEventListener && !m_hasDispatchedDisposeEvent && m_navigation;
}

void NavigationHistoryEntry::dispatchDisposeEvent()
{
    ASSERT(!m_hasDispatchedDisposeEvent);
    dispatchEvent(Event::create(eventNames().disposeEvent, { }, Event::IsTrusted::Yes));
    m_hasDispatchedDisposeEvent = true;
}

} // namespace WebCore
