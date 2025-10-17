/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#include "History.h"

#include "BackForwardController.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "FrameLoader.h"
#include "HistoryController.h"
#include "HistoryItem.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "Logging.h"
#include "Navigation.h"
#include "NavigationScheduler.h"
#include "OriginAccessPatterns.h"
#include "Page.h"
#include "Quirks.h"
#include "ScriptController.h"
#include "SecurityOrigin.h"
#include <wtf/CheckedArithmetic.h>
#include <wtf/MainThread.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

#if PLATFORM(COCOA)
#include <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#endif

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(History);

History::History(LocalDOMWindow& window)
    : LocalDOMWindowProperty(&window)
{
}

static bool isDocumentFullyActive(LocalFrame* frame)
{
    return frame && frame->protectedDocument()->isFullyActive();
}

static Exception documentNotFullyActive()
{
    return Exception { ExceptionCode::SecurityError, "Attempt to use History API from a document that isn't fully active"_s };
}

ExceptionOr<unsigned> History::length() const
{
    RefPtr frame = this->frame();
    if (!isDocumentFullyActive(frame.get()))
        return documentNotFullyActive();
    RefPtr page = frame->page();
    if (!page)
        return 0;
    return page->checkedBackForward()->count();
}

ExceptionOr<History::ScrollRestoration> History::scrollRestoration() const
{
    RefPtr frame = this->frame();
    if (!isDocumentFullyActive(frame.get()))
        return documentNotFullyActive();

    auto* historyItem = frame->loader().history().currentItem();
    if (!historyItem)
        return ScrollRestoration::Auto;
    
    return historyItem->shouldRestoreScrollPosition() ? ScrollRestoration::Auto : ScrollRestoration::Manual;
}

ExceptionOr<void> History::setScrollRestoration(ScrollRestoration scrollRestoration)
{
    RefPtr frame = this->frame();
    if (!isDocumentFullyActive(frame.get()))
        return documentNotFullyActive();

    if (RefPtr historyItem = frame->loader().history().currentItem())
        historyItem->setShouldRestoreScrollPosition(scrollRestoration == ScrollRestoration::Auto);

    return { };
}

ExceptionOr<SerializedScriptValue*> History::state()
{
    RefPtr frame = this->frame();
    if (!isDocumentFullyActive(frame.get()))
        return documentNotFullyActive();
    m_lastStateObjectRequested = stateInternal();
    return m_lastStateObjectRequested.get();
}

SerializedScriptValue* History::stateInternal() const
{
    auto* frame = this->frame();
    if (!frame)
        return nullptr;
    auto* historyItem = frame->loader().history().currentItem();
    if (!historyItem)
        return nullptr;
    return historyItem->stateObject();
}

bool History::stateChanged() const
{
    return m_lastStateObjectRequested != stateInternal();
}

JSValueInWrappedObject& History::cachedState()
{
    if (m_cachedState && stateChanged())
        m_cachedState.clear();
    return m_cachedState;
}

bool History::isSameAsCurrentState(SerializedScriptValue* state) const
{
    return state == stateInternal();
}

ExceptionOr<void> History::back()
{
    return go(-1);
}

ExceptionOr<void> History::back(Document& document)
{
    return go(document, -1);
}

ExceptionOr<void> History::forward()
{
    return go(1);
}

ExceptionOr<void> History::forward(Document& document)
{
    return go(document, 1);
}

ExceptionOr<void> History::go(int distance)
{
    RefPtr frame = this->frame();
    LOG(History, "History %p go(%d) frame %p (main frame %d)", this, distance, frame.get(), frame ? frame->isMainFrame() : false);

    if (!isDocumentFullyActive(frame.get()))
        return documentNotFullyActive();

    frame->protectedNavigationScheduler()->scheduleHistoryNavigation(distance);
    return { };
}

ExceptionOr<void> History::go(Document& document, int distance)
{
    RefPtr frame = this->frame();
    LOG(History, "History %p go(%d) in document %p frame %p (main frame %d)", this, distance, &document, frame.get(), frame ? frame->isMainFrame() : false);

    if (!isDocumentFullyActive(frame.get()))
        return documentNotFullyActive();

    ASSERT(isMainThread());

    if (!document.canNavigate(frame.get()))
        return { };

    frame->protectedNavigationScheduler()->scheduleHistoryNavigation(distance);
    return { };
}

uint32_t History::totalStateObjectPayloadLimit() const
{
    ASSERT(frame() && frame()->isMainFrame());

    // Each unique main-frame document is only allowed to send 64MB of state object payload to the UI client/process.
    static uint32_t defaultTotalStateObjectPayloadLimit = 64 * MB;
    return m_totalStateObjectPayloadLimitOverride.value_or(defaultTotalStateObjectPayloadLimit);
}

ExceptionOr<void> History::updateAndCheckStateObjectQuota(const URL& fullURL, SerializedScriptValue* data, NavigationHistoryBehavior historyBehavior)
{
    static Seconds stateObjectTimeSpan { 10_s };
    static unsigned perStateObjectTimeSpanLimit = 100;

    Checked<unsigned> urlSize = fullURL.string().length();
    urlSize *= 2;

    Checked<uint64_t> payloadSize = urlSize;
    payloadSize += data ? data->wireBytes().size() : 0;

    if (RefPtr localMainFrame = dynamicDowncast<LocalFrame>(frame()->page()->mainFrame())) {
        RefPtr mainWindow = localMainFrame->window();
        if (!mainWindow)
            return { };
        Ref mainHistory = mainWindow->history();

        WallTime currentTimestamp = WallTime::now();
        if (currentTimestamp - mainHistory->m_currentStateObjectTimeSpanStart > stateObjectTimeSpan) {
            mainHistory->m_currentStateObjectTimeSpanStart = currentTimestamp;
            mainHistory->m_currentStateObjectTimeSpanObjectsAdded = 0;
        }

        if (mainHistory->m_currentStateObjectTimeSpanObjectsAdded >= perStateObjectTimeSpanLimit) {
            if (historyBehavior == NavigationHistoryBehavior::Replace)
                return Exception { ExceptionCode::SecurityError, makeString("Attempt to use history.replaceState() more than "_s, perStateObjectTimeSpanLimit, " times per "_s, stateObjectTimeSpan.seconds(), " seconds"_s) };
            return Exception { ExceptionCode::SecurityError, makeString("Attempt to use history.pushState() more than "_s, perStateObjectTimeSpanLimit, " times per "_s, stateObjectTimeSpan.seconds(), " seconds"_s) };
        }

        Checked<uint64_t> newTotalUsage = mainHistory->m_totalStateObjectUsage;

        if (historyBehavior == NavigationHistoryBehavior::Replace)
            newTotalUsage -= m_mostRecentStateObjectUsage;
        newTotalUsage += payloadSize;

        if (newTotalUsage > mainHistory->totalStateObjectPayloadLimit()) {
            if (historyBehavior == NavigationHistoryBehavior::Replace)
                return Exception { ExceptionCode::QuotaExceededError, "Attempt to store more data than allowed using history.replaceState()"_s };
            return Exception { ExceptionCode::QuotaExceededError, "Attempt to store more data than allowed using history.pushState()"_s };
        }

        mainHistory->m_totalStateObjectUsage = newTotalUsage;
        ++mainHistory->m_currentStateObjectTimeSpanObjectsAdded;
    }

    m_mostRecentStateObjectUsage = payloadSize;
    return { };
}

// https://html.spec.whatwg.org/#shared-history-push/replace-state-steps
ExceptionOr<void> History::stateObjectAdded(RefPtr<SerializedScriptValue>&& data, const String& urlString, NavigationHistoryBehavior historyBehavior)
{
    m_cachedState.clear();

    RefPtr frame = this->frame();
    if (!isDocumentFullyActive(frame.get()))
        return documentNotFullyActive();
    if (!frame->page())
        return { };

    RefPtr document = frame->protectedDocument();
    const URL& documentURL = document->url();
    URL fullURL = documentURL;

    auto createBlockedURLSecurityErrorWithMessageSuffix = [&fullURL, documentURL, historyBehavior] (ASCIILiteral suffix) -> Exception {
        const auto functionName = historyBehavior == NavigationHistoryBehavior::Replace ? "history.replaceState()"_s : "history.pushState()"_s;
        return Exception { ExceptionCode::SecurityError, makeString("Blocked attempt to use "_s, functionName, " to change session history URL from "_s, documentURL.stringCenterEllipsizedToLength(), " to "_s, fullURL.stringCenterEllipsizedToLength(), ". "_s, suffix) };
    };

    if (!urlString.isEmpty()) {
        fullURL = document->completeURL(urlString);
        if (!fullURL.isValid())
            return createBlockedURLSecurityErrorWithMessageSuffix("URL is invalid"_s);

        // The checks below are loosely implementing https://html.spec.whatwg.org/#can-have-its-url-rewritten
        if (!protocolHostAndPortAreEqual(fullURL, documentURL) || fullURL.user() != documentURL.user() || fullURL.password() != documentURL.password())
            return createBlockedURLSecurityErrorWithMessageSuffix("Protocols, domains, ports, usernames, and passwords must match."_s);

        if (fullURL.protocolIsFile()
#if PLATFORM(COCOA)
            && linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::PushStateFilePathRestriction)
#endif
            && !document->quirks().shouldDisablePushStateFilePathRestrictions()
            && fullURL.fileSystemPath() != documentURL.fileSystemPath()) {
            return createBlockedURLSecurityErrorWithMessageSuffix("Only differences in query and fragment are allowed for file: URLs."_s);
        }

        Ref documentSecurityOrigin = frame->document()->securityOrigin();
        // We allow sandboxed documents, 'data:'/'file:' URLs, etc. to use 'pushState'/'replaceState' to modify the URL query and fragments.
        // See https://bugs.webkit.org/show_bug.cgi?id=183028 for the compatibility concerns.
        bool allowSandboxException = (documentSecurityOrigin->isLocal() || documentSecurityOrigin->isOpaque())
            && documentURL.viewWithoutQueryOrFragmentIdentifier() == fullURL.viewWithoutQueryOrFragmentIdentifier();

        if (!allowSandboxException && !documentSecurityOrigin->canRequest(fullURL, OriginAccessPatternsForWebProcess::singleton()) && (fullURL.path() != documentURL.path() || fullURL.query() != documentURL.query()))
            return createBlockedURLSecurityErrorWithMessageSuffix("Paths and fragments must match for a sandboxed document."_s);
    }

    if (auto result = updateAndCheckStateObjectQuota(fullURL, data.get(), historyBehavior); result.hasException())
        return result.releaseException();

    if (document->settings().navigationAPIEnabled()) {
        Ref navigation = document->domWindow()->navigation();
        if (!navigation->dispatchPushReplaceReloadNavigateEvent(fullURL, historyBehavior == NavigationHistoryBehavior::Push ? NavigationNavigationType::Push : NavigationNavigationType::Replace, true, nullptr, data.get()))
            return { };
    }

    frame->loader().updateURLAndHistory(fullURL, WTFMove(data), historyBehavior);
    return { };
}

} // namespace WebCore
