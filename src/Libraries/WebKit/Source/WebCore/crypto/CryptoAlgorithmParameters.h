/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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

#include "CryptoAlgorithmIdentifier.h"
#include <wtf/TypeCasts.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CryptoAlgorithmParameters {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(CryptoAlgorithmParameters, WEBCORE_EXPORT);
public:
    enum class Class : uint8_t {
        None,
        AesCbcCfbParams,
        AesCtrParams,
        AesGcmParams,
        AesKeyParams,
        EcKeyParams,
        EcdhKeyDeriveParams,
        EcdsaParams,
        HkdfParams,
        HmacKeyParams,
        Pbkdf2Params,
        RsaHashedKeyGenParams,
        RsaHashedImportParams,
        RsaKeyGenParams,
        RsaOaepParams,
        RsaPssParams,
        X25519Params,
    };

    // FIXME: Consider merging name and identifier.
    String name;
    CryptoAlgorithmIdentifier identifier;

    virtual ~CryptoAlgorithmParameters() = default;

    virtual Class parametersClass() const { return Class::None; }
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CRYPTO_ALGORITHM_PARAMETERS(ToClassName) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CryptoAlgorithm##ToClassName) \
static bool isType(const WebCore::CryptoAlgorithmParameters& parameters) { return parameters.parametersClass() == WebCore::CryptoAlgorithmParameters::Class::ToClassName; } \
SPECIALIZE_TYPE_TRAITS_END()
