/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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

#include "AdvancedPrivacyProtections.h"
#include "FrameLoaderTypes.h"
#include "ReferrerPolicy.h"
#include "ResourceRequest.h"
#include "ShouldTreatAsContinuingLoad.h"
#include "SubstituteData.h"
#include <wtf/Forward.h>

namespace WebCore {

class Document;
class LocalFrame;
class SecurityOrigin;

class FrameLoadRequest {
public:
    WEBCORE_EXPORT FrameLoadRequest(Ref<Document>&&, SecurityOrigin&, ResourceRequest&&, const AtomString& frameName, InitiatedByMainFrame, const AtomString& downloadAttribute = { });
    WEBCORE_EXPORT FrameLoadRequest(LocalFrame&, const ResourceRequest&, const SubstituteData& = SubstituteData());

    WEBCORE_EXPORT ~FrameLoadRequest();

    WEBCORE_EXPORT FrameLoadRequest(FrameLoadRequest&&);
    WEBCORE_EXPORT FrameLoadRequest& operator=(FrameLoadRequest&&);

    bool isEmpty() const { return m_resourceRequest.isEmpty(); }

    WEBCORE_EXPORT Document& requester();
    Ref<Document> protectedRequester() const;
    const SecurityOrigin& requesterSecurityOrigin() const;
    Ref<SecurityOrigin> protectedRequesterSecurityOrigin() const;

    ResourceRequest& resourceRequest() { return m_resourceRequest; }
    const ResourceRequest& resourceRequest() const { return m_resourceRequest; }

    const AtomString& frameName() const { return m_frameName; }
    void setFrameName(const AtomString& frameName) { m_frameName = frameName; }

    void setShouldCheckNewWindowPolicy(bool checkPolicy) { m_shouldCheckNewWindowPolicy = checkPolicy; }
    bool shouldCheckNewWindowPolicy() const { return m_shouldCheckNewWindowPolicy; }

    void setShouldTreatAsContinuingLoad(ShouldTreatAsContinuingLoad shouldTreatAsContinuingLoad) { m_shouldTreatAsContinuingLoad = shouldTreatAsContinuingLoad; }
    ShouldTreatAsContinuingLoad shouldTreatAsContinuingLoad() const { return m_shouldTreatAsContinuingLoad; }

    const SubstituteData& substituteData() const { return m_substituteData; }
    void setSubstituteData(const SubstituteData& data) { m_substituteData = data; }
    bool hasSubstituteData() { return m_substituteData.isValid(); }

    LockHistory lockHistory() const { return m_lockHistory; }
    void setLockHistory(LockHistory value) { m_lockHistory = value; }

    LockBackForwardList lockBackForwardList() const { return m_lockBackForwardList; }
    void setLockBackForwardList(LockBackForwardList value) { m_lockBackForwardList = value; }

    bool isInitialFrameSrcLoad() const { return m_isInitialFrameSrcLoad; }
    void setIsInitialFrameSrcLoad(bool isInitialFrameSrcLoad) { m_isInitialFrameSrcLoad = isInitialFrameSrcLoad; }

    const String& clientRedirectSourceForHistory() const { return m_clientRedirectSourceForHistory; }
    void setClientRedirectSourceForHistory(const String& clientRedirectSourceForHistory) { m_clientRedirectSourceForHistory = clientRedirectSourceForHistory; }

    ReferrerPolicy referrerPolicy() const { return m_referrerPolicy; }
    void setReferrerPolicy(const ReferrerPolicy& referrerPolicy) { m_referrerPolicy = referrerPolicy; }

    AllowNavigationToInvalidURL allowNavigationToInvalidURL() const { return m_allowNavigationToInvalidURL; }
    void disableNavigationToInvalidURL() { m_allowNavigationToInvalidURL = AllowNavigationToInvalidURL::No; }

    NewFrameOpenerPolicy newFrameOpenerPolicy() const { return m_newFrameOpenerPolicy; }
    void setNewFrameOpenerPolicy(NewFrameOpenerPolicy newFrameOpenerPolicy) { m_newFrameOpenerPolicy = newFrameOpenerPolicy; }

    // The shouldReplaceDocumentIfJavaScriptURL parameter will go away when the FIXME to eliminate the
    // corresponding parameter from ScriptController::executeIfJavaScriptURL() is addressed.
    ShouldReplaceDocumentIfJavaScriptURL shouldReplaceDocumentIfJavaScriptURL() const { return m_shouldReplaceDocumentIfJavaScriptURL; }
    void disableShouldReplaceDocumentIfJavaScriptURL() { m_shouldReplaceDocumentIfJavaScriptURL = DoNotReplaceDocumentIfJavaScriptURL; }

    void setShouldOpenExternalURLsPolicy(ShouldOpenExternalURLsPolicy policy) { m_shouldOpenExternalURLsPolicy = policy; }
    ShouldOpenExternalURLsPolicy shouldOpenExternalURLsPolicy() const { return m_shouldOpenExternalURLsPolicy; }

    const AtomString& downloadAttribute() const { return m_downloadAttribute; }

    InitiatedByMainFrame initiatedByMainFrame() const { return m_initiatedByMainFrame; }

    void setIsRequestFromClientOrUserInput() { m_isRequestFromClientOrUserInput = true; }
    bool isRequestFromClientOrUserInput() const { return m_isRequestFromClientOrUserInput; }

    void setAdvancedPrivacyProtections(OptionSet<AdvancedPrivacyProtections> policy) { m_advancedPrivacyProtections = policy; }
    std::optional<OptionSet<AdvancedPrivacyProtections>> advancedPrivacyProtections() const { return m_advancedPrivacyProtections; }

    NavigationHistoryBehavior navigationHistoryBehavior() const { return m_navigationHistoryBehavior; }
    void setNavigationHistoryBehavior(NavigationHistoryBehavior historyHandling) { m_navigationHistoryBehavior = historyHandling; }

    bool isFromNavigationAPI() const { return m_isFromNavigationAPI; }
    void setIsFromNavigationAPI(bool isFromNavigationAPI) { m_isFromNavigationAPI = isFromNavigationAPI; }

private:
    Ref<Document> m_requester;
    Ref<SecurityOrigin> m_requesterSecurityOrigin;
    ResourceRequest m_resourceRequest;
    AtomString m_frameName;
    SubstituteData m_substituteData;
    String m_clientRedirectSourceForHistory;

    bool m_shouldCheckNewWindowPolicy { false };
    ShouldTreatAsContinuingLoad m_shouldTreatAsContinuingLoad { ShouldTreatAsContinuingLoad::No };
    LockHistory m_lockHistory { LockHistory::No };
    LockBackForwardList m_lockBackForwardList { LockBackForwardList::No };
    ReferrerPolicy m_referrerPolicy { ReferrerPolicy::EmptyString };
    AllowNavigationToInvalidURL m_allowNavigationToInvalidURL { AllowNavigationToInvalidURL::Yes };
    NewFrameOpenerPolicy m_newFrameOpenerPolicy { NewFrameOpenerPolicy::Allow };
    ShouldReplaceDocumentIfJavaScriptURL m_shouldReplaceDocumentIfJavaScriptURL { ReplaceDocumentIfJavaScriptURL };
    ShouldOpenExternalURLsPolicy m_shouldOpenExternalURLsPolicy { ShouldOpenExternalURLsPolicy::ShouldNotAllow };
    AtomString m_downloadAttribute;
    InitiatedByMainFrame m_initiatedByMainFrame { InitiatedByMainFrame::Unknown };
    bool m_isRequestFromClientOrUserInput { false };
    bool m_isInitialFrameSrcLoad { false };
    std::optional<OptionSet<AdvancedPrivacyProtections>> m_advancedPrivacyProtections;
    NavigationHistoryBehavior m_navigationHistoryBehavior { NavigationHistoryBehavior::Auto };
    bool m_isFromNavigationAPI { false };
};

} // namespace WebCore
