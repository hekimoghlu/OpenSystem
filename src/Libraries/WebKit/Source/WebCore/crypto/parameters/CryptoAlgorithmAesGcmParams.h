/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include <wtf/Vector.h>

namespace WebCore {

class CryptoAlgorithmAesGcmParams final : public CryptoAlgorithmParameters {
    WTF_MAKE_TZONE_ALLOCATED(CryptoAlgorithmAesGcmParams);
public:
    BufferSource iv;
    // Use additionalDataVector() instead of additionalData. The label will be gone once additionalDataVector() is called.
    mutable std::optional<BufferSource::VariantType> additionalData;
    mutable std::optional<uint8_t> tagLength;

    Class parametersClass() const final { return Class::AesGcmParams; }

    const Vector<uint8_t>& ivVector() const
    {
        if (!m_ivVector.isEmpty() || !iv.length())
            return m_ivVector;

        m_ivVector.append(iv.span());
        return m_ivVector;
    }

    const Vector<uint8_t>& additionalDataVector() const
    {
        if (!m_additionalDataVector.isEmpty() || !additionalData)
            return m_additionalDataVector;

        BufferSource additionalDataBuffer = WTFMove(*additionalData);
        additionalData = std::nullopt;
        if (!additionalDataBuffer.length())
            return m_additionalDataVector;

        m_additionalDataVector.append(additionalDataBuffer.span());
        return m_additionalDataVector;
    }

    CryptoAlgorithmAesGcmParams isolatedCopy() const
    {
        CryptoAlgorithmAesGcmParams result;
        result.identifier = identifier;
        result.m_ivVector = ivVector();
        result.m_additionalDataVector = additionalDataVector();
        result.tagLength = tagLength;

        return result;
    }

private:
    mutable Vector<uint8_t> m_ivVector;
    mutable Vector<uint8_t> m_additionalDataVector;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_ALGORITHM_PARAMETERS(AesGcmParams)
