/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

constexpr auto ed25519KeySize = 32;
constexpr auto ed25519SignatureSize = ed25519KeySize * 2;

class CryptoKeyOKP;

class CryptoAlgorithmEd25519 final : public CryptoAlgorithm {
public:
    static constexpr ASCIILiteral s_name = "Ed25519"_s;
    static constexpr CryptoAlgorithmIdentifier s_identifier = CryptoAlgorithmIdentifier::Ed25519;
    static Ref<CryptoAlgorithm> create();

private:
    CryptoAlgorithmEd25519() = default;
    CryptoAlgorithmIdentifier identifier() const final;

    void generateKey(const CryptoAlgorithmParameters& , bool extractable, CryptoKeyUsageBitmap usages, KeyOrKeyPairCallback&& , ExceptionCallback&& , ScriptExecutionContext&);
    void sign(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&) final;
    void verify(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&& signature, Vector<uint8_t>&&, BoolCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&) final;
    void importKey(CryptoKeyFormat, KeyData&&, const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyCallback&&, ExceptionCallback&&) final;
    void exportKey(CryptoKeyFormat, Ref<CryptoKey>&&, KeyDataCallback&&, ExceptionCallback&&) final;

    static ExceptionOr<Vector<uint8_t>> platformSign(const CryptoKeyOKP&, const Vector<uint8_t>&);
    static ExceptionOr<bool> platformVerify(const CryptoKeyOKP&, const Vector<uint8_t>&, const Vector<uint8_t>&);
};

inline CryptoAlgorithmIdentifier CryptoAlgorithmEd25519::identifier() const
{
    return s_identifier;
}

} // namespace WebCore
