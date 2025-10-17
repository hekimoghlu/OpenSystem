/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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

class CryptoAlgorithmPbkdf2Params;
class CryptoKeyRaw;

class CryptoAlgorithmPBKDF2 final : public CryptoAlgorithm {
public:
    static constexpr ASCIILiteral s_name = "PBKDF2"_s;
    static constexpr CryptoAlgorithmIdentifier s_identifier = CryptoAlgorithmIdentifier::PBKDF2;
    static Ref<CryptoAlgorithm> create();

private:
    CryptoAlgorithmPBKDF2() = default;
    CryptoAlgorithmIdentifier identifier() const final;

    void deriveBits(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, std::optional<size_t> length, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&) final;
    void importKey(CryptoKeyFormat, KeyData&&, const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyCallback&&, ExceptionCallback&&) final;
    ExceptionOr<std::optional<size_t>> getKeyLength(const CryptoAlgorithmParameters&) final;

    static ExceptionOr<Vector<uint8_t>> platformDeriveBits(const CryptoAlgorithmPbkdf2Params&, const CryptoKeyRaw&, size_t);
};

} // namespace WebCore
