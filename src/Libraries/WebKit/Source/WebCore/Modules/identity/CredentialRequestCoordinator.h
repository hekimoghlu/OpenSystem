/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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

#include "CredentialRequestCoordinatorClient.h"
#include "IDLTypes.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AbortSignal;
class BasicCredential;
class CredentialRequestCoordinatorClient;
class Document;
struct CredentialRequestOptions;

template<typename IDLType> class DOMPromiseDeferred;

using CredentialPromise = DOMPromiseDeferred<IDLNullable<IDLInterface<BasicCredential>>>;

class CredentialRequestCoordinator final {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(CredentialRequestCoordinator, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(CredentialRequestCoordinator);

public:
    WEBCORE_EXPORT explicit CredentialRequestCoordinator(std::unique_ptr<CredentialRequestCoordinatorClient>&&);
    void discoverFromExternalSource(const Document&, CredentialRequestOptions&&, CredentialPromise&&);

private:
    CredentialRequestCoordinator() = default;

    std::unique_ptr<CredentialRequestCoordinatorClient> m_client;
    bool m_isCancelling = false;
    CompletionHandler<void()> m_queuedRequest;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
