/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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

#include "IdentifierTypes.h"
#include "WebPage.h"
#include <WebCore/ScrollTypes.h>
#include <WebCore/VisibleSelection.h>

#if ENABLE(APP_HIGHLIGHTS)
#include <WebCore/AppHighlight.h>
#endif

namespace WebKit {

struct WebPage::Internals {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
#if PLATFORM(IOS_FAMILY)
    WebCore::VisibleSelection storedSelectionForAccessibility { WebCore::VisibleSelection() };
    FocusedElementInformationIdentifier lastFocusedElementInformationIdentifier;
    TransactionID lastTransactionIDWithScaleChange;
    std::optional<std::pair<TransactionID, double>> lastLayerTreeTransactionIdAndPageScaleBeforeScalingPage;
#endif
#if ENABLE(APP_HIGHLIGHTS)
    WebCore::CreateNewGroupForHighlight highlightIsNewGroup { WebCore::CreateNewGroupForHighlight::No };
    WebCore::HighlightRequestOriginatedInApp highlightRequestOriginatedInApp { WebCore::HighlightRequestOriginatedInApp::No };
#endif
    std::optional<WebsitePoliciesData> pendingWebsitePolicies;
    WebCore::ScrollPinningBehavior scrollPinningBehavior { WebCore::ScrollPinningBehavior::DoNotPin };
    mutable EditorStateIdentifier lastEditorStateIdentifier;
    HashMap<WebCore::RegistrableDomain, HashSet<WebCore::RegistrableDomain>> domainsWithPageLevelStorageAccess;
    HashSet<WebCore::RegistrableDomain> loadedSubresourceDomains;
#if ENABLE(ADVANCED_PRIVACY_PROTECTIONS)
    struct LinkDecorationFilteringConditionals {
        HashSet<WebCore::RegistrableDomain> domains;
        Vector<String> paths;
    };
    HashMap<String, LinkDecorationFilteringConditionals> linkDecorationFilteringData;
    HashMap<WebCore::RegistrableDomain, HashSet<String>> allowedQueryParametersForAdvancedPrivacyProtections;
#endif
};

} // namespace WebKit
