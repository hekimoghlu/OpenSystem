/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

#include "EventTarget.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalDOMWindowProperty.h"
#include "NavigateEvent.h"
#include "NavigationHistoryEntry.h"
#include "NavigationNavigationType.h"
#include "NavigationTransition.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class FormState;
class HistoryItem;
class SerializedScriptValue;
class NavigateEvent;
class NavigationActivation;
class NavigationDestination;

enum class FrameLoadType : uint8_t;

enum class NavigationAPIMethodTrackerType { };
using NavigationAPIMethodTrackerIdentifier = ObjectIdentifier<NavigationAPIMethodTrackerType>;

// https://html.spec.whatwg.org/multipage/nav-history-apis.html#navigation-api-method-tracker
struct NavigationAPIMethodTracker : public RefCounted<NavigationAPIMethodTracker> {
    WTF_MAKE_STRUCT_FAST_ALLOCATED(NavigationAPIMethodTracker);

    static Ref<NavigationAPIMethodTracker> create(Ref<DeferredPromise>&& committed, Ref<DeferredPromise>&& finished, JSC::JSValue&& info, RefPtr<SerializedScriptValue>&& serializedState)
    {
        return adoptRef(*new NavigationAPIMethodTracker(WTFMove(committed), WTFMove(finished), WTFMove(info), WTFMove(serializedState)));
    }

    bool operator==(const NavigationAPIMethodTracker& other) const
    {
        // key is optional so we manually identify each tracker.
        return identifier == other.identifier;
    }

    bool finishedBeforeCommit { false };
    String key;
    JSC::JSValue info;
    RefPtr<SerializedScriptValue> serializedState;
    RefPtr<NavigationHistoryEntry> committedToEntry;
    Ref<DeferredPromise> committedPromise;
    Ref<DeferredPromise> finishedPromise;

private:
    explicit NavigationAPIMethodTracker(Ref<DeferredPromise>&& committed, Ref<DeferredPromise>&& finished, JSC::JSValue&& info, RefPtr<SerializedScriptValue>&& serializedState)
        : info(info)
        , serializedState(serializedState)
        , committedPromise(WTFMove(committed))
        , finishedPromise(WTFMove(finished))
        , identifier(NavigationAPIMethodTrackerIdentifier::generate())
    {
    }

    NavigationAPIMethodTrackerIdentifier identifier;
};

enum class ShouldCopyStateObjectFromCurrentEntry : bool { No, Yes };

class Navigation final : public RefCounted<Navigation>, public EventTarget, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Navigation);
public:
    ~Navigation();

    static Ref<Navigation> create(LocalDOMWindow& window) { return adoptRef(*new Navigation(window)); }

    using RefCounted<Navigation>::ref;
    using RefCounted<Navigation>::deref;

    using HistoryBehavior = NavigationHistoryBehavior;

    struct UpdateCurrentEntryOptions {
        JSC::JSValue state;
    };

    struct Options {
        JSC::JSValue info;
    };

    struct NavigateOptions : Options {
        JSC::JSValue state;
        HistoryBehavior history;
    };

    struct ReloadOptions : Options {
        JSC::JSValue state;
    };

    struct Result {
        RefPtr<DOMPromise> committed;
        RefPtr<DOMPromise> finished;
    };

    const Vector<Ref<NavigationHistoryEntry>>& entries() const;
    NavigationHistoryEntry* currentEntry() const;
    NavigationTransition* transition() { return m_transition.get(); };
    NavigationActivation* activation() { return m_activation.get(); };

    bool canGoBack() const;
    bool canGoForward() const;

    void initializeForNewWindow(std::optional<NavigationNavigationType>, LocalDOMWindow* previousWindow);

    Result navigate(const String& url, NavigateOptions&&, Ref<DeferredPromise>&&, Ref<DeferredPromise>&&);

    Result reload(ReloadOptions&&, Ref<DeferredPromise>&&, Ref<DeferredPromise>&&);

    Result traverseTo(const String& key, Options&&, Ref<DeferredPromise>&&, Ref<DeferredPromise>&&);
    Result back(Options&&, Ref<DeferredPromise>&&, Ref<DeferredPromise>&&);
    Result forward(Options&&, Ref<DeferredPromise>&&, Ref<DeferredPromise>&&);

    ExceptionOr<void> updateCurrentEntry(UpdateCurrentEntryOptions&&);

    enum class DispatchResult : uint8_t { Completed, Aborted, Intercepted };
    DispatchResult dispatchTraversalNavigateEvent(HistoryItem&);
    bool dispatchPushReplaceReloadNavigateEvent(const URL&, NavigationNavigationType, bool isSameDocument, FormState*, SerializedScriptValue* classicHistoryAPIState = nullptr);
    bool dispatchDownloadNavigateEvent(const URL&, const String& downloadFilename);

    void updateForNavigation(Ref<HistoryItem>&&, NavigationNavigationType, ShouldCopyStateObjectFromCurrentEntry = ShouldCopyStateObjectFromCurrentEntry::No);
    void updateForReactivation(Vector<Ref<HistoryItem>>& newHistoryItems, HistoryItem& reactivatedItem);
    void updateForActivation(HistoryItem* previousItem, std::optional<NavigationNavigationType>);

    RefPtr<NavigationActivation> createForPageswapEvent(HistoryItem* newItem, DocumentLoader*, bool fromBackForwardCache);

    void abortOngoingNavigationIfNeeded();

    std::optional<Ref<NavigationHistoryEntry>> findEntryByKey(const String& key);
    bool suppressNormalScrollRestoration() const { return m_suppressNormalScrollRestorationDuringOngoingNavigation; }

    void setFocusChanged(FocusDidChange changed) { m_focusChangedDuringOngoingNavigation = changed; }

    // EventTarget.
    ScriptExecutionContext* scriptExecutionContext() const final;
    RefPtr<ScriptExecutionContext> protectedScriptExecutionContext() const;

    void rejectFinishedPromise(NavigationAPIMethodTracker*);
    NavigationAPIMethodTracker* upcomingTraverseMethodTracker(const String& key) const { return m_upcomingTraverseMethodTrackers.get(key); }

    class AbortHandler : public RefCountedAndCanMakeWeakPtr<AbortHandler> {
    public:
        bool wasAborted() const { return m_wasAborted; }

    private:
        friend class Navigation;

        static Ref<AbortHandler> create() { return adoptRef(*new AbortHandler); }
        void markAsAborted() { m_wasAborted = true; }

        bool m_wasAborted { false };
    };
    Ref<AbortHandler> registerAbortHandler();

private:
    explicit Navigation(LocalDOMWindow&);

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final;
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    bool hasEntriesAndEventsDisabled() const;
    Result performTraversal(const String& key, Navigation::Options, Ref<DeferredPromise>&& committed, Ref<DeferredPromise>&& finished);
    ExceptionOr<RefPtr<SerializedScriptValue>> serializeState(JSC::JSValue state);
    DispatchResult innerDispatchNavigateEvent(NavigationNavigationType, Ref<NavigationDestination>&&, const String& downloadRequestFilename, FormState* = nullptr, SerializedScriptValue* classicHistoryAPIState = nullptr);

    RefPtr<NavigationAPIMethodTracker> maybeSetUpcomingNonTraversalTracker(Ref<DeferredPromise>&& committed, Ref<DeferredPromise>&& finished, JSC::JSValue info, RefPtr<SerializedScriptValue>&&);
    RefPtr<NavigationAPIMethodTracker> addUpcomingTraverseAPIMethodTracker(Ref<DeferredPromise>&& committed, Ref<DeferredPromise>&& finished, const String& key, JSC::JSValue info);
    void cleanupAPIMethodTracker(NavigationAPIMethodTracker*);
    void resolveFinishedPromise(NavigationAPIMethodTracker*);
    void rejectFinishedPromise(NavigationAPIMethodTracker*, const Exception&, JSC::JSValue exceptionObject);
    void abortOngoingNavigation(NavigateEvent&);
    void promoteUpcomingAPIMethodTracker(const String& destinationKey);
    void notifyCommittedToEntry(NavigationAPIMethodTracker*, NavigationHistoryEntry*, NavigationNavigationType);
    Result apiMethodTrackerDerivedResult(const NavigationAPIMethodTracker&);

    std::optional<size_t> m_currentEntryIndex;
    RefPtr<NavigationTransition> m_transition;
    RefPtr<NavigationActivation> m_activation;
    Vector<Ref<NavigationHistoryEntry>> m_entries;

    RefPtr<NavigateEvent> m_ongoingNavigateEvent;
    FocusDidChange m_focusChangedDuringOngoingNavigation { FocusDidChange::No };
    bool m_suppressNormalScrollRestorationDuringOngoingNavigation { false };
    RefPtr<NavigationAPIMethodTracker> m_ongoingAPIMethodTracker;
    RefPtr<NavigationAPIMethodTracker> m_upcomingNonTraverseMethodTracker;
    HashMap<String, Ref<NavigationAPIMethodTracker>> m_upcomingTraverseMethodTrackers;
    WeakHashSet<AbortHandler> m_abortHandlers;
};

} // namespace WebCore
