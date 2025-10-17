/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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

class CryptoAlgorithmSHA224 final : public CryptoAlgorithm {
public:
    static constexpr ASCIILiteral s_name = "SHA-224"_s;
    static constexpr CryptoAlgorithmIdentifier s_identifier = CryptoAlgorithmIdentifier::SHA_224;
    static Ref<CryptoAlgorithm> create();

private:
    CryptoAlgorithmSHA224() = default;
    CryptoAlgorithmIdentifier identifier() const final;
    void digest(Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&) final;
};

} // namespace WebCore
