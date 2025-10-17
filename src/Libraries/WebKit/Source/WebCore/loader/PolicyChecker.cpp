/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#include "PolicyChecker.h"

#include "BlobRegistry.h"
#include "ContentFilter.h"
#include "ContentSecurityPolicy.h"
#include "DocumentInlines.h"
#include "DocumentLoader.h"
#include "Event.h"
#include "EventHandler.h"
#include "EventLoop.h"
#include "EventNames.h"
#include "FormState.h"
#include "FrameLoader.h"
#include "HTMLFormElement.h"
#include "HTMLFrameOwnerElement.h"
#include "HTMLPlugInElement.h"
#include "HitTestResult.h"
#include "LoaderStrategy.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "Logging.h"
#include "Navigation.h"
#include "PlatformStrategies.h"
#include "ThreadableBlobRegistry.h"
#include "URLKeepingBlobAlive.h"
#include <wtf/CompletionHandler.h>

#if USE(QUICK_LOOK)
#include "QuickLook.h"
#endif

#define PAGE_ID (m_frame->pageID() ? m_frame->pageID()->toUInt64() : 0)
#define FRAME_ID (m_frame->loader().frameID().object().toUInt64())
#define POLICYCHECKER_RELEASE_LOG(fmt, ...) RELEASE_LOG(Loading, "%p - [pageID=%" PRIu64 ", frameID=%" PRIu64 "] PolicyChecker::" fmt, this, PAGE_ID, FRAME_ID, ##__VA_ARGS__)

namespace WebCore {

static bool isAllowedByContentSecurityPolicy(const URL& url, const Element* ownerElement, bool didReceiveRedirectResponse)
{
    if (!ownerElement)
        return true;
    // Elements in user agent show tree should load whatever the embedding document policy is.
    if (ownerElement->isInUserAgentShadowTree())
        return true;

    auto redirectResponseReceived = didReceiveRedirectResponse ? ContentSecurityPolicy::RedirectResponseReceived::Yes : ContentSecurityPolicy::RedirectResponseReceived::No;

    ASSERT(ownerElement->document().contentSecurityPolicy());
    if (is<HTMLPlugInElement>(ownerElement))
        return ownerElement->protectedDocument()->checkedContentSecurityPolicy()->allowObjectFromSource(url, redirectResponseReceived);
    return ownerElement->protectedDocument()->checkedContentSecurityPolicy()->allowChildFrameFromSource(url, redirectResponseReceived);
}

static bool shouldExecuteJavaScriptURLSynchronously(const URL& url)
{
    // FIXME: Some sites rely on the javascript:'' loading synchronously, which is why we have this special case.
    // Blink has the same workaround (https://bugs.chromium.org/p/chromium/issues/detail?id=923585).
    return url == "javascript:''"_s || url == "javascript:\"\""_s;
}

PolicyChecker::PolicyChecker(LocalFrame& frame)
    : m_frame(frame)
    , m_delegateIsDecidingNavigationPolicy(false)
    , m_delegateIsHandlingUnimplementablePolicy(false)
    , m_loadType(FrameLoadType::Standard)
{
}

void PolicyChecker::checkNavigationPolicy(ResourceRequest&& newRequest, const ResourceResponse& redirectResponse, NavigationPolicyDecisionFunction&& function)
{
    checkNavigationPolicy(WTFMove(newRequest), redirectResponse, m_frame->loader().protectedActiveDocumentLoader().get(), { }, WTFMove(function));
}

URLKeepingBlobAlive PolicyChecker::extendBlobURLLifetimeIfNecessary(const ResourceRequest& request, const Document& document, PolicyDecisionMode mode) const
{
    if (mode != PolicyDecisionMode::Asynchronous || !request.url().protocolIsBlob())
        return { };

    bool haveTriggeringRequester = m_frame->loader().policyDocumentLoader() && !m_frame->loader().policyDocumentLoader()->triggeringAction().isEmpty() && m_frame->loader().policyDocumentLoader()->triggeringAction().requester();
    auto& topOrigin = haveTriggeringRequester ? m_frame->loader().policyDocumentLoader()->triggeringAction().requester()->topOrigin->data() : document.topOrigin().data();
    return { request.url(), topOrigin };
}

void PolicyChecker::checkNavigationPolicy(ResourceRequest&& request, const ResourceResponse& redirectResponse, DocumentLoader* loader, RefPtr<FormState>&& formState, NavigationPolicyDecisionFunction&& function, PolicyDecisionMode policyDecisionMode)
{
    NavigationAction action = loader->triggeringAction();
    Ref frame = m_frame.get();
    if (action.isEmpty()) {
        action = NavigationAction { frame->protectedDocument().releaseNonNull(), request, InitiatedByMainFrame::Unknown, loader->isRequestFromClientOrUserInput(), NavigationType::Other, loader->shouldOpenExternalURLsPolicyToPropagate() };
        loader->setTriggeringAction(NavigationAction { action });
    }

    Ref frameLoader = frame->loader();
    if (frame->page() && frame->page()->openedByDOMWithOpener())
        action.setOpenedByDOMWithOpener();
    action.setHasOpenedFrames(frameLoader->hasOpenedFrames());

    // Don't ask more than once for the same request or if we are loading an empty URL.
    // This avoids confusion on the part of the client.
    if (equalIgnoringHeaderFields(request, loader->lastCheckedRequest()) || (!request.isNull() && request.url().isEmpty())) {
        if (!request.isNull() && request.url().isEmpty())
            POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: continuing because the URL is empty");
        else
            POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: continuing because the URL is the same as the last request");
        function(ResourceRequest(request), { }, NavigationPolicyDecision::ContinueLoad);
        loader->setLastCheckedRequest(WTFMove(request));
        return;
    }

    // We are always willing to show alternate content for unreachable URLs;
    // treat it like a reload so it maintains the right state for b/f list.
    auto& substituteData = loader->substituteData();
    if (substituteData.isValid() && !substituteData.failingURL().isEmpty()) {
        bool shouldContinue = true;
#if ENABLE(CONTENT_FILTERING)
        if (RefPtr loader = frameLoader->activeDocumentLoader())
            shouldContinue = ContentFilter::continueAfterSubstituteDataRequest(*loader, substituteData);
#endif
        if (isBackForwardLoadType(m_loadType))
            m_loadType = FrameLoadType::Reload;
        if (shouldContinue)
            POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: continuing because we have valid substitute data");
        else
            POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: not continuing with substitute data because the content filter told us not to");

        function(WTFMove(request), { }, shouldContinue ? NavigationPolicyDecision::ContinueLoad : NavigationPolicyDecision::IgnoreLoad);
        return;
    }

    RefPtr frameOwnerElement = frame->ownerElement();
    if (!isAllowedByContentSecurityPolicy(request.url(), frameOwnerElement.get(), !redirectResponse.isNull())) {
        if (frameOwnerElement) {
            // Fire a load event (even though we were blocked by CSP) as timing attacks would otherwise
            // reveal that the frame was blocked. This way, it looks like any other cross-origin page load.
            frameOwnerElement->dispatchEvent(Event::create(eventNames().loadEvent, Event::CanBubble::No, Event::IsCancelable::No));
        }
        POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: ignoring because disallowed by content security policy");
        function(WTFMove(request), { }, NavigationPolicyDecision::IgnoreLoad);
        return;
    }

    loader->setLastCheckedRequest(ResourceRequest(request));

    ASSERT(frameOwnerElement == m_frame->ownerElement());

    // Only the PDFDocument iframe is allowed to navigate to webkit-pdfjs-viewer URLs
    bool isInPDFDocumentFrame = frameOwnerElement && frameOwnerElement->document().isPDFDocument();
    if (isInPDFDocumentFrame && request.url().protocolIs("webkit-pdfjs-viewer"_s)) {
        POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: continuing because PDFJS URL");
        return function(WTFMove(request), formState, NavigationPolicyDecision::ContinueLoad);
    }

#if USE(QUICK_LOOK)
    // Always allow QuickLook-generated URLs based on the protocol scheme.
    if (!request.isNull() && isQuickLookPreviewURL(request.url())) {
        POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: continuing because quicklook-generated URL");
        return function(WTFMove(request), formState, NavigationPolicyDecision::ContinueLoad);
    }
#endif

#if ENABLE(CONTENT_FILTERING)
    if (m_contentFilterUnblockHandler.canHandleRequest(request)) {
        m_contentFilterUnblockHandler.requestUnblockAsync([frame](bool unblocked) {
            if (unblocked)
                frame->protectedLoader()->reload();
        });
        POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: ignoring because ContentFilterUnblockHandler can handle the request");
        return function({ }, nullptr, NavigationPolicyDecision::IgnoreLoad);
    }
    m_contentFilterUnblockHandler = { };
#endif

    frameLoader->clearProvisionalLoadForPolicyCheck();

    auto blobURLLifetimeExtension = extendBlobURLLifetimeIfNecessary(request, *frame->protectedDocument(), policyDecisionMode);
    bool requestIsJavaScriptURL = request.url().protocolIsJavaScript();
    bool isInitialEmptyDocumentLoad = !frameLoader->stateMachine().committedFirstRealDocumentLoad() && request.url().protocolIsAbout() && !substituteData.isValid();
    m_delegateIsDecidingNavigationPolicy = true;
    String suggestedFilename = action.downloadAttribute().isEmpty() ? nullAtom() : action.downloadAttribute();
    FromDownloadAttribute fromDownloadAttribute = action.downloadAttribute().isNull() ? FromDownloadAttribute::No : FromDownloadAttribute::Yes;
    if (!action.downloadAttribute().isNull()) {
        RefPtr document = frame->document();
        if (document && document->settings().navigationAPIEnabled()) {
            if (RefPtr domWindow = document->domWindow()) {
                if (!domWindow->protectedNavigation()->dispatchDownloadNavigateEvent(request.url(), action.downloadAttribute()))
                    return function({ }, nullptr, NavigationPolicyDecision::IgnoreLoad);
            }
        }
    }
    FramePolicyFunction decisionHandler = [
        this,
        weakThis = WeakPtr { *this },
        function = WTFMove(function),
        request = ResourceRequest(request),
        requestIsJavaScriptURL,
        formState = std::exchange(formState, nullptr),
        suggestedFilename = WTFMove(suggestedFilename),
        blobURLLifetimeExtension = WTFMove(blobURLLifetimeExtension),
        isInitialEmptyDocumentLoad,
        fromDownloadAttribute
    ] (PolicyAction policyAction) mutable {
        if (!weakThis)
            return function({ }, nullptr, NavigationPolicyDecision::IgnoreLoad);

        m_delegateIsDecidingNavigationPolicy = false;

        Ref frame = m_frame.get();
        Ref frameLoader = frame->loader();
        switch (policyAction) {
        case PolicyAction::Download:
            if (!frame->effectiveSandboxFlags().contains(SandboxFlag::Downloads)) {
                frameLoader->setOriginalURLForDownloadRequest(request);
                frameLoader->client().startDownload(request, suggestedFilename, fromDownloadAttribute);
            } else if (RefPtr document = frame->document())
                document->addConsoleMessage(MessageSource::Security, MessageLevel::Error, "Not allowed to download due to sandboxing"_s);
            FALLTHROUGH;
        case PolicyAction::Ignore:
            POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: ignoring because policyAction from dispatchDecidePolicyForNavigationAction is Ignore");
            return function({ }, nullptr, NavigationPolicyDecision::IgnoreLoad);
        case PolicyAction::LoadWillContinueInAnotherProcess:
            POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: stopping because policyAction from dispatchDecidePolicyForNavigationAction is LoadWillContinueInAnotherProcess");
            function({ }, nullptr, NavigationPolicyDecision::LoadWillContinueInAnotherProcess);
            return;
        case PolicyAction::Use:
            if (!requestIsJavaScriptURL && !frameLoader->client().canHandleRequest(request)) {
                handleUnimplementablePolicy(platformStrategies()->loaderStrategy()->cannotShowURLError(request));
                POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: ignoring because frame loader client can't handle the request");
                return function({ }, { }, NavigationPolicyDecision::IgnoreLoad);
            }
            if (isInitialEmptyDocumentLoad)
                POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: continuing because this is an initial empty document");
            else
                POLICYCHECKER_RELEASE_LOG("checkNavigationPolicy: continuing because this policyAction from dispatchDecidePolicyForNavigationAction is Use");
            return function(WTFMove(request), formState, NavigationPolicyDecision::ContinueLoad);
        }
        ASSERT_NOT_REACHED();
    };

    // Historically, we haven't asked the client for JS URL but we still want to execute them asynchronously.
    if (requestIsJavaScriptURL) {
        RefPtr document = frame->document();
        if (!document)
            return decisionHandler(PolicyAction::Ignore);

        if (shouldExecuteJavaScriptURLSynchronously(request.url()))
            return decisionHandler(PolicyAction::Use);

        document->checkedEventLoop()->queueTask(TaskSource::DOMManipulation, [weakThis = WeakPtr { *this }, decisionHandler = WTFMove(decisionHandler), identifier = m_javaScriptURLPolicyCheckIdentifier] () mutable {
            if (!weakThis)
                return decisionHandler(PolicyAction::Ignore);
            // Don't proceed if PolicyChecker::stopCheck has been called between the call to queueTask and now.
            if (weakThis->m_javaScriptURLPolicyCheckIdentifier != identifier)
                return decisionHandler(PolicyAction::Ignore);
            decisionHandler(PolicyAction::Use);
        });
        return;
    }

    auto documentLoader = frameLoader->loaderForWebsitePolicies();
    auto clientRedirectSourceForHistory = documentLoader ? documentLoader->clientRedirectSourceForHistory() : String();
    auto navigationID = documentLoader ? documentLoader->navigationID() : std::nullopt;
    bool hasOpener = !!frame->opener();
    auto sandboxFlags = frame->effectiveSandboxFlags();
    auto isPerformingHTTPFallback = frameLoader->isHTTPFallbackInProgress() ? IsPerformingHTTPFallback::Yes : IsPerformingHTTPFallback::No;

    if (isInitialEmptyDocumentLoad) {
        // We ignore the response from the client for initial empty document loads and proceed with the load synchronously.
        frameLoader->client().dispatchDecidePolicyForNavigationAction(action, request, redirectResponse, formState.get(), clientRedirectSourceForHistory, navigationID, hitTestResult(action), hasOpener, isPerformingHTTPFallback, sandboxFlags, policyDecisionMode, [](PolicyAction) { });
        decisionHandler(PolicyAction::Use);
    } else
        frameLoader->client().dispatchDecidePolicyForNavigationAction(action, request, redirectResponse, formState.get(), clientRedirectSourceForHistory, navigationID, hitTestResult(action), hasOpener, isPerformingHTTPFallback, sandboxFlags, policyDecisionMode, WTFMove(decisionHandler));
}

Ref<LocalFrame> PolicyChecker::protectedFrame() const
{
    return m_frame.get();
}

std::optional<HitTestResult> PolicyChecker::hitTestResult(const NavigationAction& action)
{
    auto& mouseEventData = action.mouseEventData();
    if (!mouseEventData)
        return std::nullopt;
    constexpr OptionSet<HitTestRequest::Type> hitType { HitTestRequest::Type::ReadOnly, HitTestRequest::Type::Active, HitTestRequest::Type::DisallowUserAgentShadowContent, HitTestRequest::Type::AllowChildFrameContent };
    return protectedFrame()->checkedEventHandler()->hitTestResultAtPoint(mouseEventData->absoluteLocation, hitType);
}

void PolicyChecker::checkNewWindowPolicy(NavigationAction&& navigationAction, ResourceRequest&& request, RefPtr<FormState>&& formState, const AtomString& frameName, NewWindowPolicyDecisionFunction&& function)
{
    if (m_frame->document() && m_frame->document()->isSandboxed(SandboxFlag::Popups))
        return function({ }, nullptr, { }, { }, ShouldContinuePolicyCheck::No);

    if (!LocalDOMWindow::allowPopUp(m_frame))
        return function({ }, nullptr, { }, { }, ShouldContinuePolicyCheck::No);

    auto blobURLLifetimeExtension = extendBlobURLLifetimeIfNecessary(request, *m_frame->document());

    Ref frame = m_frame.get();
    frame->protectedLoader()->client().dispatchDecidePolicyForNewWindowAction(navigationAction, request, formState.get(), frameName, hitTestResult(navigationAction), [frame, request,
        formState = WTFMove(formState), frameName, navigationAction, function = WTFMove(function), blobURLLifetimeExtension = WTFMove(blobURLLifetimeExtension)] (PolicyAction policyAction) mutable {

        switch (policyAction) {
        case PolicyAction::Download:
            if (!frame->effectiveSandboxFlags().contains(SandboxFlag::Downloads))
                frame->protectedLoader()->client().startDownload(request);
            else if (RefPtr document = frame->document())
                document->addConsoleMessage(MessageSource::Security, MessageLevel::Error, "Not allowed to download due to sandboxing"_s);
            FALLTHROUGH;
        case PolicyAction::Ignore:
            function({ }, nullptr, { }, { }, ShouldContinuePolicyCheck::No);
            return;
        case PolicyAction::LoadWillContinueInAnotherProcess:
            ASSERT_NOT_REACHED();
            function({ }, nullptr, { }, { }, ShouldContinuePolicyCheck::No);
            return;
        case PolicyAction::Use:
            function(request, formState, frameName, navigationAction, ShouldContinuePolicyCheck::Yes);
            return;
        }
        ASSERT_NOT_REACHED();
    });
}

void PolicyChecker::stopCheck()
{
    m_javaScriptURLPolicyCheckIdentifier++;
    protectedFrame()->protectedLoader()->client().cancelPolicyCheck();
}

void PolicyChecker::cannotShowMIMEType(const ResourceResponse& response)
{
    handleUnimplementablePolicy(platformStrategies()->loaderStrategy()->cannotShowMIMETypeError(response));
}

void PolicyChecker::handleUnimplementablePolicy(const ResourceError& error)
{
    m_delegateIsHandlingUnimplementablePolicy = true;
    protectedFrame()->protectedLoader()->client().dispatchUnableToImplementPolicy(error);
    m_delegateIsHandlingUnimplementablePolicy = false;
}

} // namespace WebCore

#undef IS_ALLOWED
#undef PAGE_ID
#undef FRAME_ID
#undef POLICYCHECKER_RELEASE_LOG
