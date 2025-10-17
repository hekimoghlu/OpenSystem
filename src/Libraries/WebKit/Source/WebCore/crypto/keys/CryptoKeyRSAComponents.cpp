/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
#include "config.h"
#include "CryptoKeyRSAComponents.h"

namespace WebCore {

CryptoKeyRSAComponents::CryptoKeyRSAComponents(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent)
    : m_type(Type::Public)
    , m_modulus(modulus)
    , m_exponent(exponent)
{
}

CryptoKeyRSAComponents::CryptoKeyRSAComponents(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent)
    : m_type(Type::Public)
    , m_modulus(WTFMove(modulus))
    , m_exponent(WTFMove(exponent))
{
}

CryptoKeyRSAComponents::CryptoKeyRSAComponents(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent, const Vector<uint8_t>& privateExponent)
    : m_type(Type::Private)
    , m_modulus(modulus)
    , m_exponent(exponent)
    , m_privateExponent(privateExponent)
    , m_hasAdditionalPrivateKeyParameters(false)
{
}

CryptoKeyRSAComponents::CryptoKeyRSAComponents(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent, Vector<uint8_t>&& privateExponent)
    : m_type(Type::Private)
    , m_modulus(WTFMove(modulus))
    , m_exponent(WTFMove(exponent))
    , m_privateExponent(WTFMove(privateExponent))
    , m_hasAdditionalPrivateKeyParameters(false)
{
}

CryptoKeyRSAComponents::CryptoKeyRSAComponents(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent, const Vector<uint8_t>& privateExponent, const PrimeInfo& firstPrimeInfo, const PrimeInfo& secondPrimeInfo, const Vector<PrimeInfo>& otherPrimeInfos)
    : m_type(Type::Private)
    , m_modulus(modulus)
    , m_exponent(exponent)
    , m_privateExponent(privateExponent)
    , m_hasAdditionalPrivateKeyParameters(true)
    , m_firstPrimeInfo(firstPrimeInfo)
    , m_secondPrimeInfo(secondPrimeInfo)
    , m_otherPrimeInfos(otherPrimeInfos)
{
}

CryptoKeyRSAComponents::CryptoKeyRSAComponents(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent, Vector<uint8_t>&& privateExponent, PrimeInfo&& firstPrimeInfo, PrimeInfo&& secondPrimeInfo, Vector<PrimeInfo>&& otherPrimeInfos)
    : m_type(Type::Private)
    , m_modulus(WTFMove(modulus))
    , m_exponent(WTFMove(exponent))
    , m_privateExponent(WTFMove(privateExponent))
    , m_hasAdditionalPrivateKeyParameters(true)
    , m_firstPrimeInfo(WTFMove(firstPrimeInfo))
    , m_secondPrimeInfo(WTFMove(secondPrimeInfo))
    , m_otherPrimeInfos(WTFMove(otherPrimeInfos))
{
}

CryptoKeyRSAComponents::~CryptoKeyRSAComponents() = default;

} // namespace WebCore
