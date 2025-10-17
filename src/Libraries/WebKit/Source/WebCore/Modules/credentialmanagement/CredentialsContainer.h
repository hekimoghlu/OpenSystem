/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

#include "AuthenticatorCoordinator.h"
#include "CredentialRequestCoordinator.h"
#include "DigitalCredential.h"
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class WeakPtrImplWithEventTargetData;
struct CredentialCreationOptions;
struct CredentialRequestOptions;

class CredentialsContainer : public RefCounted<CredentialsContainer> {
public:
    static Ref<CredentialsContainer> create(WeakPtr<Document, WeakPtrImplWithEventTargetData>&& document)
    {
        return adoptRef(*new CredentialsContainer(WTFMove(document)));
    }

    virtual void get(CredentialRequestOptions&&, CredentialPromise&&);

    void store(const BasicCredential&, CredentialPromise&&);

    virtual void isCreate(CredentialCreationOptions&&, CredentialPromise&&);

    void preventSilentAccess(DOMPromiseDeferred<void>&&) const;

    CredentialsContainer(WeakPtr<Document, WeakPtrImplWithEventTargetData>&&);

    virtual ~CredentialsContainer() = default;

private:
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;

protected:
    template<typename Options>
    bool performCommonChecks(const Options&, CredentialPromise&);
    const Document* document() const { return m_document.get(); }
};

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
