/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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

#if ENABLE(WEB_AUTHN)

#include "APIWebAuthenticationPanel.h"
#include "FrameInfoData.h"
#include "WebAuthenticationFlags.h"
#include "WebPageProxy.h"
#include <WebCore/CredentialRequestOptions.h>
#include <WebCore/GlobalFrameIdentifier.h>
#include <WebCore/PublicKeyCredentialCreationOptions.h>
#include <WebCore/PublicKeyCredentialRequestOptions.h>
#include <WebCore/WebAuthenticationConstants.h>
#include <variant>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {

class WebPageProxy;

struct WebAuthenticationRequestData {
    Vector<uint8_t> hash;
    std::variant<WebCore::PublicKeyCredentialCreationOptions, WebCore::PublicKeyCredentialRequestOptions> options;

    // FIXME<rdar://problem/71509848>: Remove the following deprecated fields.
    WeakPtr<WebPageProxy> page;
    WebAuthenticationPanelResult panelResult { WebAuthenticationPanelResult::Unavailable };
    RefPtr<API::WebAuthenticationPanel> panel;
    std::optional<WebCore::GlobalFrameIdentifier> globalFrameID;
    WebKit::FrameInfoData frameInfo;

    String cachedPin; // Only used to improve NFC Client PIN experience.
    WeakPtr<API::WebAuthenticationPanel> weakPanel;
    std::optional<WebCore::MediationRequirement> mediation;
    std::optional<WebCore::SecurityOriginData> parentOrigin;
};

WebCore::ClientDataType getClientDataType(const std::variant<WebCore::PublicKeyCredentialCreationOptions, WebCore::PublicKeyCredentialRequestOptions>&);
WebCore::UserVerificationRequirement getUserVerificationRequirement(const std::variant<WebCore::PublicKeyCredentialCreationOptions, WebCore::PublicKeyCredentialRequestOptions>&);

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
