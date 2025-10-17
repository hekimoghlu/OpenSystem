/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

class CryptoAlgorithmRsaOaepParams;
class CryptoKeyRSA;

class CryptoAlgorithmRSA_OAEP final : public CryptoAlgorithm {
public:
    static constexpr ASCIILiteral s_name = "RSA-OAEP"_s;
    static constexpr CryptoAlgorithmIdentifier s_identifier = CryptoAlgorithmIdentifier::RSA_OAEP;
    static Ref<CryptoAlgorithm> create();

private:
    CryptoAlgorithmRSA_OAEP() = default;
    CryptoAlgorithmIdentifier identifier() const final;

    void encrypt(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&) final;
    void decrypt(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&) final;
    void generateKey(const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyOrKeyPairCallback&&, ExceptionCallback&&, ScriptExecutionContext&) final;
    void importKey(CryptoKeyFormat, KeyData&&, const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyCallback&&, ExceptionCallback&&) final;
    void exportKey(CryptoKeyFormat, Ref<CryptoKey>&&, KeyDataCallback&&, ExceptionCallback&&) final;

    static ExceptionOr<Vector<uint8_t>> platformEncrypt(const CryptoAlgorithmRsaOaepParams&, const CryptoKeyRSA&, const Vector<uint8_t>&);
    static ExceptionOr<Vector<uint8_t>> platformDecrypt(const CryptoAlgorithmRsaOaepParams&, const CryptoKeyRSA&, const Vector<uint8_t>&);
};

} // namespace WebCore
