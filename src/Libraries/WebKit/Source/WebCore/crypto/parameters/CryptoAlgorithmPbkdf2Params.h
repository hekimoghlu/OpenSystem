/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#include "BufferSource.h"
#include "CryptoAlgorithmParameters.h"
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/Strong.h>
#include <wtf/Vector.h>

namespace WebCore {

class CryptoAlgorithmPbkdf2Params final : public CryptoAlgorithmParameters {
    WTF_MAKE_TZONE_ALLOCATED(CryptoAlgorithmPbkdf2Params);
public:
    BufferSource salt;
    unsigned long iterations;
    // FIXME: Consider merging hash and hashIdentifier.
    std::variant<JSC::Strong<JSC::JSObject>, String> hash;
    CryptoAlgorithmIdentifier hashIdentifier;

    const Vector<uint8_t>& saltVector() const
    {
        if (!m_saltVector.isEmpty() || !salt.length())
            return m_saltVector;

        m_saltVector.append(salt.span());
        return m_saltVector;
    }

    Class parametersClass() const final { return Class::Pbkdf2Params; }

    CryptoAlgorithmPbkdf2Params isolatedCopy() const
    {
        CryptoAlgorithmPbkdf2Params result;
        result.identifier = identifier;
        result.m_saltVector = saltVector();
        result.iterations = iterations;
        result.hashIdentifier = hashIdentifier;

        return result;
    }

private:
    mutable Vector<uint8_t> m_saltVector;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_ALGORITHM_PARAMETERS(Pbkdf2Params)
