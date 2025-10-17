/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include "IdentityCredentialsContainer.h"

#if ENABLE(WEB_AUTHN)

#include "CredentialCreationOptions.h"
#include "CredentialRequestOptions.h"
#include "DigitalCredential.h"
#include "DigitalCredentialRequestOptions.h"
#include "Document.h"
#include "ExceptionOr.h"
#include "JSDOMPromiseDeferred.h"
#include "JSDigitalCredential.h"
#include "LocalDOMWindow.h"
#include "MediationRequirement.h"
#include "Page.h"
#include "VisibilityState.h"

namespace WebCore {
IdentityCredentialsContainer::IdentityCredentialsContainer(WeakPtr<Document, WeakPtrImplWithEventTargetData>&& document)
    : CredentialsContainer(WTFMove(document))
{
}

void IdentityCredentialsContainer::get(CredentialRequestOptions&& options, CredentialPromise&& promise)
{
    if (!performCommonChecks(options, promise))
        return;

    if (!options.digital || options.publicKey) {
        promise.reject(Exception { ExceptionCode::NotSupportedError, "Only digital member is supported."_s });
        return;
    }

    if (options.mediation != MediationRequirement::Required) {
        promise.reject(Exception { ExceptionCode::TypeError, "User mediation is required for DigitalCredential."_s });
        return;
    }

    RefPtr document = this->document();
    ASSERT(document);

    if (!PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::DigitalCredentialsGetRule, *document, PermissionsPolicy::ShouldReportViolation::No)) {
        promise.reject(Exception { ExceptionCode::NotAllowedError, "Third-party iframes are not allowed to call .get() unless explicitly allowed via Permissions Policy (digital-credentials-get)"_s });
        return;
    }

    if (options.digital->requests.isEmpty()) {
        promise.reject(Exception { ExceptionCode::TypeError, "At least one request must present."_s });
        return;
    }

    RefPtr window = document->domWindow();
    if (!window || !window->consumeTransientActivation()) {
        promise.reject(Exception { ExceptionCode::NotAllowedError, "Calling get() needs to be triggered by an activation triggering user event."_s });
        return;
    }

    document->page()->credentialRequestCoordinator().discoverFromExternalSource(*document, WTFMove(options), WTFMove(promise));
}

void IdentityCredentialsContainer::isCreate(CredentialCreationOptions&& options, CredentialPromise&& promise)
{
    if (!performCommonChecks(options, promise))
        return;

    // Default as per Cred Man spec is to resolve with null.
    // https://www.w3.org/TR/credential-management-1/#algorithm-create-cred
    promise.resolve(nullptr);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
