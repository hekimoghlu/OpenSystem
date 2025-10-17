/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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

#include "CryptoAlgorithmParameters.h"
#include <JavaScriptCore/Uint8Array.h>
#include <wtf/Vector.h>

namespace WebCore {

class CryptoAlgorithmRsaKeyGenParams : public CryptoAlgorithmParameters {
    WTF_MAKE_TZONE_ALLOCATED(CryptoAlgorithmRsaKeyGenParams);
public:
    size_t modulusLength;
    RefPtr<Uint8Array> publicExponent;

    Class parametersClass() const override { return Class::RsaKeyGenParams; }

    const Vector<uint8_t>& publicExponentVector() const
    {
        if (!m_publicExponentVector.isEmpty() || !publicExponent->byteLength())
            return m_publicExponentVector;

        m_publicExponentVector.append(publicExponent->span());
        return m_publicExponentVector;
    }
private:
    mutable Vector<uint8_t> m_publicExponentVector;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_ALGORITHM_PARAMETERS(RsaKeyGenParams)
