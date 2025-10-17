/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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

#if HAVE(RSA_PSS)

namespace WebCore {

class CryptoAlgorithmRsaPssParams;
class CryptoKeyRSA;

class CryptoAlgorithmRSA_PSS final : public CryptoAlgorithm {
public:
    static constexpr ASCIILiteral s_name = "RSA-PSS"_s;
    static constexpr CryptoAlgorithmIdentifier s_identifier = CryptoAlgorithmIdentifier::RSA_PSS;
    static Ref<CryptoAlgorithm> create();

private:
    CryptoAlgorithmRSA_PSS() = default;
    CryptoAlgorithmIdentifier identifier() const final;

    void sign(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&) final;
    void verify(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&& signature, Vector<uint8_t>&&, BoolCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&) final;
    void generateKey(const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyOrKeyPairCallback&&, ExceptionCallback&&, ScriptExecutionContext&) final;
    void importKey(CryptoKeyFormat, KeyData&&, const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyCallback&&, ExceptionCallback&&) final;
    void exportKey(CryptoKeyFormat, Ref<CryptoKey>&&, KeyDataCallback&&, ExceptionCallback&&) final;

    static ExceptionOr<Vector<uint8_t>> platformSign(const CryptoAlgorithmRsaPssParams&, const CryptoKeyRSA&, const Vector<uint8_t>&);
    static ExceptionOr<bool> platformVerify(const CryptoAlgorithmRsaPssParams&, const CryptoKeyRSA&, const Vector<uint8_t>&, const Vector<uint8_t>&);
};

} // namespace WebCore

#endif // HAVE(RSA_PSS)
