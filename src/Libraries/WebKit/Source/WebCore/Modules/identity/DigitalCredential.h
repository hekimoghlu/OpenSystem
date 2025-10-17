/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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

#include "BasicCredential.h"
#include "IDLTypes.h"
#include "IdentityCredentialProtocol.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class DigitalCredential;
template<typename IDLType> class DOMPromiseDeferred;

using DigitalCredentialPromise = DOMPromiseDeferred<IDLInterface<DigitalCredential>>;

class DigitalCredential final : public BasicCredential {
public:
    static Ref<DigitalCredential> create(JSC::Strong<JSC::JSObject>&&, IdentityCredentialProtocol);

    virtual ~DigitalCredential();

    const JSC::Strong<JSC::JSObject>& data() const
    {
        return m_data;
    };

    IdentityCredentialProtocol protocol() const
    {
        return m_protocol;
    }

private:
    DigitalCredential(JSC::Strong<JSC::JSObject>&&, IdentityCredentialProtocol);

    Type credentialType() const final { return Type::DigitalCredential; }

    IdentityCredentialProtocol m_protocol;
    const JSC::Strong<JSC::JSObject> m_data;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BASIC_CREDENTIAL(DigitalCredential, BasicCredential::Type::DigitalCredential)

#endif // ENABLE(WEB_AUTHN)
