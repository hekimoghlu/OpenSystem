/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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

#include "CryptoAlgorithm.h"

namespace WebCore {

class CryptoKeyAES;

class CryptoAlgorithmAESKW final : public CryptoAlgorithm {
public:
    static constexpr ASCIILiteral s_name = "AES-KW"_s;
    static constexpr CryptoAlgorithmIdentifier s_identifier = CryptoAlgorithmIdentifier::AES_KW;
    static Ref<CryptoAlgorithm> create();
    static constexpr size_t s_extraSize = 64 / 8; // https://datatracker.ietf.org/doc/html/rfc3394#section-2.2.1

private:
    CryptoAlgorithmAESKW() = default;
    CryptoAlgorithmIdentifier identifier() const final;

    void generateKey(const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyOrKeyPairCallback&&, ExceptionCallback&&, ScriptExecutionContext&) final;
    void importKey(CryptoKeyFormat, KeyData&&, const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyCallback&&, ExceptionCallback&&) final;
    void exportKey(CryptoKeyFormat, Ref<CryptoKey>&&, KeyDataCallback&&, ExceptionCallback&&) final;
    void wrapKey(Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&) final;
    void unwrapKey(Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&) final;
    ExceptionOr<std::optional<size_t>> getKeyLength(const CryptoAlgorithmParameters&) final;

    static ExceptionOr<Vector<uint8_t>> platformWrapKey(const CryptoKeyAES&, const Vector<uint8_t>&);
    static ExceptionOr<Vector<uint8_t>> platformUnwrapKey(const CryptoKeyAES&, const Vector<uint8_t>&);
};

} // namespace WebCore
