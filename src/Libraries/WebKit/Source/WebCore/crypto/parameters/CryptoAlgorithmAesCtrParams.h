/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

class CryptoAlgorithmAesCtrParams final : public CryptoAlgorithmParameters {
    WTF_MAKE_TZONE_ALLOCATED(CryptoAlgorithmAesCtrParams);
public:
    BufferSource counter;
    size_t length;

    Class parametersClass() const final { return Class::AesCtrParams; }

    const Vector<uint8_t>& counterVector() const
    {
        if (!m_counterVector.isEmpty() || !counter.length())
            return m_counterVector;

        m_counterVector.append(counter.span());
        return m_counterVector;
    }

    CryptoAlgorithmAesCtrParams isolatedCopy() const
    {
        CryptoAlgorithmAesCtrParams result;
        result.identifier = identifier;
        result.m_counterVector = counterVector();
        result.length = length;

        return result;
    }

private:
    mutable Vector<uint8_t> m_counterVector;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_ALGORITHM_PARAMETERS(AesCtrParams)
