/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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

class CryptoAlgorithmHkdfParams final : public CryptoAlgorithmParameters {
    WTF_MAKE_TZONE_ALLOCATED(CryptoAlgorithmHkdfParams);
public:
    // FIXME: Consider merging hash and hashIdentifier.
    std::variant<JSC::Strong<JSC::JSObject>, String> hash;
    CryptoAlgorithmIdentifier hashIdentifier;
    BufferSource salt;
    BufferSource info;

    const Vector<uint8_t>& saltVector() const
    {
        if (!m_saltVector.isEmpty() || !salt.length())
            return m_saltVector;

        m_saltVector.append(salt.span());
        return m_saltVector;
    }

    const Vector<uint8_t>& infoVector() const
    {
        if (!m_infoVector.isEmpty() || !info.length())
            return m_infoVector;

        m_infoVector.append(info.span());
        return m_infoVector;
    }

    Class parametersClass() const final { return Class::HkdfParams; }

    CryptoAlgorithmHkdfParams isolatedCopy() const
    {
        CryptoAlgorithmHkdfParams result;
        result.identifier = identifier;
        result.m_saltVector = saltVector();
        result.m_infoVector = infoVector();
        result.hashIdentifier = hashIdentifier;

        return result;
    }

private:
    mutable Vector<uint8_t> m_saltVector;
    mutable Vector<uint8_t> m_infoVector;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_ALGORITHM_PARAMETERS(HkdfParams)
