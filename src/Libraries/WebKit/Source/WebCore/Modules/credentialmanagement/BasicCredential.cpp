/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
#include "BasicCredential.h"

#if ENABLE(WEB_AUTHN)

#include "AuthenticatorCoordinator.h"
#include "JSDOMPromiseDeferred.h"
#include "Page.h"

namespace WebCore {

BasicCredential::BasicCredential(const String& id, Type type, Discovery discovery)
    : m_id(id)
    , m_type(type)
    , m_discovery(discovery)
{
}

BasicCredential::~BasicCredential() = default;

String BasicCredential::type() const
{
    switch (m_type) {
    case Type::DigitalCredential:
        return "digital-credential"_s;

    case Type::PublicKey:
        return "public-key"_s;
    }

    ASSERT_NOT_REACHED();
    return emptyString();
}

void BasicCredential::isConditionalMediationAvailable(Document& document, DOMPromiseDeferred<IDLBoolean>&& promise)
{
    if (RefPtr page = document.page())
        page->authenticatorCoordinator().isConditionalMediationAvailable(document, WTFMove(promise));
    else
        promise.reject(Exception { ExceptionCode::InvalidStateError });
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
