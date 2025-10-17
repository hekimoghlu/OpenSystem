/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#include "CredentialsContainer.h"

#if ENABLE(WEB_AUTHN)

#include "AbortSignal.h"
#include "CredentialCreationOptions.h"
#include "CredentialRequestCoordinator.h"
#include "CredentialRequestOptions.h"
#include "DigitalCredential.h"
#include "DigitalCredentialRequestOptions.h"
#include "Document.h"
#include "ExceptionOr.h"
#include "JSDOMPromiseDeferred.h"
#include "JSDigitalCredential.h"
#include "Page.h"
#include "SecurityOrigin.h"

namespace WebCore {

CredentialsContainer::CredentialsContainer(WeakPtr<Document, WeakPtrImplWithEventTargetData>&& document)
    : m_document(WTFMove(document))
{
}

void CredentialsContainer::get(CredentialRequestOptions&& options, CredentialPromise&& promise)
{
    // The following implements https://www.w3.org/TR/credential-management-1/#algorithm-request as of 4 August 2017
    // with enhancement from 14 November 2017 Editor's Draft.
    if (!performCommonChecks(options, promise)) {
        return;
    }

    if (options.digital) {
        document()->page()->credentialRequestCoordinator().discoverFromExternalSource(*document(), WTFMove(options), WTFMove(promise));
        return;
    }

    document()->page()->authenticatorCoordinator().discoverFromExternalSource(*document(), WTFMove(options), WTFMove(promise));
}

void CredentialsContainer::store(const BasicCredential&, CredentialPromise&& promise)
{
    promise.reject(Exception { ExceptionCode::NotSupportedError, "Not implemented."_s });
}

void CredentialsContainer::isCreate(CredentialCreationOptions&& options, CredentialPromise&& promise)
{
    if (!performCommonChecks(options, promise))
        return;

    // Extra.
    if (!document()->hasFocus()) {
        promise.reject(Exception { ExceptionCode::NotAllowedError, "The document is not focused."_s });
        return;
    }

    if (options.publicKey) {
        document()->page()->authenticatorCoordinator().create(*document(), WTFMove(options), WTFMove(options.signal), WTFMove(promise));
        return;
    }

    promise.resolve(nullptr);
}

void CredentialsContainer::preventSilentAccess(DOMPromiseDeferred<void>&& promise) const
{
    if (RefPtr document = this->document(); !document->isFullyActive()) {
        promise.reject(Exception { ExceptionCode::InvalidStateError, "The document is not fully active."_s });
        return;
    }
    promise.resolve();
}

template<typename Options>
bool CredentialsContainer::performCommonChecks(const Options& options, CredentialPromise& promise)
{
    RefPtr document = this->document();
    if (!document) {
        promise.reject(Exception { ExceptionCode::NotSupportedError });
        return false;
    }

    if (!document->isFullyActive()) {
        promise.reject(Exception { ExceptionCode::InvalidStateError, "The document is not fully active."_s });
        return false;
    }

    if (!document->page()) {
        promise.reject(Exception { ExceptionCode::InvalidStateError, "No browsing context"_s });
        return false;
    }

    if (options.signal && options.signal->aborted()) {
        promise.rejectType<IDLAny>(options.signal->reason().getValue());
        return false;
    }

    if constexpr (std::is_same_v<Options, CredentialRequestOptions>) {
        if (!options.publicKey && !options.digital) {
            promise.reject(Exception { ExceptionCode::NotSupportedError, "Missing request type."_s });
            return false;
        }

        if (options.publicKey && options.digital) {
            promise.reject(Exception { ExceptionCode::NotSupportedError, "Only one request type is supported at a time."_s });
            return false;
        }
    }

    ASSERT(document->isSecureContext());
    return true;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
