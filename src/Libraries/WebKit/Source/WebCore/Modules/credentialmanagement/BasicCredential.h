/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#include "Document.h"
#include "IDLTypes.h"
#include "JSDOMPromiseDeferredForward.h"
#include <wtf/RefCounted.h>
#include <wtf/TypeCasts.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class BasicCredential : public RefCounted<BasicCredential> {
public:
    enum class Type {
        DigitalCredential,
        PublicKey,
    };

    enum class Discovery {
        CredentialStore,
        Remote,
    };

    BasicCredential(const String&, Type, Discovery);
    virtual ~BasicCredential();

    virtual Type credentialType() const = 0;

    const String& id() const { return m_id; }
    String type() const;
    Discovery discovery() const { return m_discovery; }

    static void isConditionalMediationAvailable(Document&, DOMPromiseDeferred<IDLBoolean>&&);

private:
    String m_id;
    Type m_type;
    Discovery m_discovery;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_BASIC_CREDENTIAL(ToClassName, Type) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToClassName) \
    static bool isType(const WebCore::BasicCredential& credential) { return credential.credentialType() == WebCore::Type; } \
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_AUTHN)
