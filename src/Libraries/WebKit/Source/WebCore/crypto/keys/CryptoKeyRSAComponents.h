/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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

#include <wtf/Vector.h>

namespace WebCore {

class CryptoKeyRSAComponents {
public:
    enum class Type : bool {
        Public,
        Private
    };

    struct PrimeInfo {
        Vector<uint8_t> primeFactor;
        Vector<uint8_t> factorCRTExponent;
        Vector<uint8_t> factorCRTCoefficient;
    };

    static std::unique_ptr<CryptoKeyRSAComponents> createPublic(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent)
    {
        return std::unique_ptr<CryptoKeyRSAComponents>(new CryptoKeyRSAComponents(modulus, exponent));
    }
    static std::unique_ptr<CryptoKeyRSAComponents> createPublic(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent)
    {
        return std::unique_ptr<CryptoKeyRSAComponents>(new CryptoKeyRSAComponents(WTFMove(modulus), WTFMove(exponent)));
    }

    static std::unique_ptr<CryptoKeyRSAComponents> createPrivate(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent, const Vector<uint8_t>& privateExponent)
    {
        return std::unique_ptr<CryptoKeyRSAComponents>(new CryptoKeyRSAComponents(modulus, exponent, privateExponent));
    }
    static std::unique_ptr<CryptoKeyRSAComponents> createPrivate(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent, Vector<uint8_t>&& privateExponent)
    {
        return std::unique_ptr<CryptoKeyRSAComponents>(new CryptoKeyRSAComponents(WTFMove(modulus), WTFMove(exponent), WTFMove(privateExponent)));
    }

    static std::unique_ptr<CryptoKeyRSAComponents> createPrivateWithAdditionalData(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent, const Vector<uint8_t>& privateExponent, const PrimeInfo& firstPrimeInfo, const PrimeInfo& secondPrimeInfo, const Vector<PrimeInfo>& otherPrimeInfos)
    {
        return std::unique_ptr<CryptoKeyRSAComponents>(new CryptoKeyRSAComponents(modulus, exponent, privateExponent, firstPrimeInfo, secondPrimeInfo, otherPrimeInfos));
    }
    static std::unique_ptr<CryptoKeyRSAComponents> createPrivateWithAdditionalData(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent, Vector<uint8_t>&& privateExponent, PrimeInfo&& firstPrimeInfo, PrimeInfo&& secondPrimeInfo, Vector<PrimeInfo>&& otherPrimeInfos)
    {
        return std::unique_ptr<CryptoKeyRSAComponents>(new CryptoKeyRSAComponents(WTFMove(modulus), WTFMove(exponent), WTFMove(privateExponent), WTFMove(firstPrimeInfo), WTFMove(secondPrimeInfo), WTFMove(otherPrimeInfos)));
    }

    virtual ~CryptoKeyRSAComponents();

    Type type() const { return m_type; }

    // Private and public keys.
    const Vector<uint8_t>& modulus() const { return m_modulus; }
    const Vector<uint8_t>& exponent() const { return m_exponent; }

    // Only private keys.
    const Vector<uint8_t>& privateExponent() const { return m_privateExponent; }
    bool hasAdditionalPrivateKeyParameters() const { return m_hasAdditionalPrivateKeyParameters; }
    const PrimeInfo& firstPrimeInfo() const { return m_firstPrimeInfo; }
    const PrimeInfo& secondPrimeInfo() const { return m_secondPrimeInfo; }
    const Vector<PrimeInfo>& otherPrimeInfos() const { return m_otherPrimeInfos; }

private:
    CryptoKeyRSAComponents(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent);
    CryptoKeyRSAComponents(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent);

    CryptoKeyRSAComponents(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent, const Vector<uint8_t>& privateExponent);
    CryptoKeyRSAComponents(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent, Vector<uint8_t>&& privateExponent);

    CryptoKeyRSAComponents(const Vector<uint8_t>& modulus, const Vector<uint8_t>& exponent, const Vector<uint8_t>& privateExponent, const PrimeInfo& firstPrimeInfo, const PrimeInfo& secondPrimeInfo, const Vector<PrimeInfo>& otherPrimeInfos);
    CryptoKeyRSAComponents(Vector<uint8_t>&& modulus, Vector<uint8_t>&& exponent, Vector<uint8_t>&& privateExponent, PrimeInfo&& firstPrimeInfo, PrimeInfo&& secondPrimeInfo, Vector<PrimeInfo>&& otherPrimeInfos);

    Type m_type;

    // Private and public keys.
    Vector<uint8_t> m_modulus;
    Vector<uint8_t> m_exponent;

    // Only private keys.
    Vector<uint8_t> m_privateExponent;
    bool m_hasAdditionalPrivateKeyParameters;
    PrimeInfo m_firstPrimeInfo;
    PrimeInfo m_secondPrimeInfo;
    Vector<PrimeInfo> m_otherPrimeInfos; // When three or more primes have been used, the number of array elements is be the number of primes used minus two.
};

} // namespace WebCore
