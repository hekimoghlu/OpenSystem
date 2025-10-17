/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

#include "CryptoAesKeyAlgorithm.h"
#include "CryptoEcKeyAlgorithm.h"
#include "CryptoHmacKeyAlgorithm.h"
#include "CryptoKeyAlgorithm.h"
#include "CryptoKeyData.h"
#include "CryptoRsaHashedKeyAlgorithm.h"
#include "CryptoRsaKeyAlgorithm.h"
#include <variant>
#include <wtf/Forward.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/TypeCasts.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class WebCoreOpaqueRoot;

class CryptoKey : public ThreadSafeRefCounted<CryptoKey> {
public:
    using Type = CryptoKeyType;
    using Data = CryptoKeyData;
    using KeyAlgorithm = std::variant<CryptoKeyAlgorithm, CryptoAesKeyAlgorithm, CryptoEcKeyAlgorithm, CryptoHmacKeyAlgorithm, CryptoRsaHashedKeyAlgorithm, CryptoRsaKeyAlgorithm>;

    CryptoKey(CryptoAlgorithmIdentifier, Type, bool extractable, CryptoKeyUsageBitmap);
    virtual ~CryptoKey();

    Type type() const;
    bool extractable() const { return m_extractable; }
    Vector<CryptoKeyUsage> usages() const;
    virtual KeyAlgorithm algorithm() const = 0;
    virtual CryptoKeyClass keyClass() const = 0;
    virtual bool isValid() const { return true; }

    WEBCORE_EXPORT virtual Data data() const = 0;
    WEBCORE_EXPORT static RefPtr<CryptoKey> create(Data&&);

    CryptoAlgorithmIdentifier algorithmIdentifier() const { return m_algorithmIdentifier; }
    CryptoKeyUsageBitmap usagesBitmap() const { return m_usages; }
    void setUsagesBitmap(CryptoKeyUsageBitmap usage) { m_usages = usage; };
    bool allows(CryptoKeyUsageBitmap usage) const { return usage == (m_usages & usage); }

    static Vector<uint8_t> randomData(size_t);

private:
    CryptoAlgorithmIdentifier m_algorithmIdentifier;
    Type m_type;
    bool m_extractable;
    CryptoKeyUsageBitmap m_usages;
};

inline auto CryptoKey::type() const -> Type
{
    return m_type;
}

WebCoreOpaqueRoot root(CryptoKey*);

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CRYPTO_KEY(ToClassName, KeyClass) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToClassName) \
    static bool isType(const WebCore::CryptoKey& key) { return key.keyClass() == WebCore::KeyClass; } \
SPECIALIZE_TYPE_TRAITS_END()
