/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#include "CryptoAlgorithmAESCTR.h"

#include "CryptoAlgorithmAesCtrParams.h"
#include "CryptoAlgorithmAesKeyParams.h"
#include "CryptoKeyAES.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/FlipBytes.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

namespace CryptoAlgorithmAESCTRInternal {
static constexpr auto ALG128 = "A128CTR"_s;
static constexpr auto ALG192 = "A192CTR"_s;
static constexpr auto ALG256 = "A256CTR"_s;
static constexpr size_t counterSize = 16;
static constexpr uint64_t allBitsSet = ~(uint64_t)0;
}

static inline bool usagesAreInvalidForCryptoAlgorithmAESCTR(CryptoKeyUsageBitmap usages)
{
    return usages & (CryptoKeyUsageSign | CryptoKeyUsageVerify | CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits);
}

static bool parametersAreValid(const CryptoAlgorithmAesCtrParams& parameters)
{
    using namespace CryptoAlgorithmAESCTRInternal;
    if (parameters.counterVector().size() != counterSize)
        return false;
    if (!parameters.length || parameters.length > 128)
        return false;
    return true;
}

Ref<CryptoAlgorithm> CryptoAlgorithmAESCTR::create()
{
    return adoptRef(*new CryptoAlgorithmAESCTR);
}

CryptoAlgorithmIdentifier CryptoAlgorithmAESCTR::identifier() const
{
    return s_identifier;
}

void CryptoAlgorithmAESCTR::encrypt(const CryptoAlgorithmParameters& parameters, Ref<CryptoKey>&& key, Vector<uint8_t>&& plainText, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    auto& aesParameters = downcast<CryptoAlgorithmAesCtrParams>(parameters);
    if (!parametersAreValid(aesParameters)) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [parameters = crossThreadCopy(aesParameters), key = WTFMove(key), plainText = WTFMove(plainText)] {
            return platformEncrypt(parameters, downcast<CryptoKeyAES>(key.get()), plainText);
        });
}

void CryptoAlgorithmAESCTR::decrypt(const CryptoAlgorithmParameters& parameters, Ref<CryptoKey>&& key, Vector<uint8_t>&& cipherText, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    auto& aesParameters = downcast<CryptoAlgorithmAesCtrParams>(parameters);
    if (!parametersAreValid(aesParameters)) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [parameters = crossThreadCopy(aesParameters), key = WTFMove(key), cipherText = WTFMove(cipherText)] {
            return platformDecrypt(parameters, downcast<CryptoKeyAES>(key.get()), cipherText);
        });
}

void CryptoAlgorithmAESCTR::generateKey(const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyOrKeyPairCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&)
{
    const auto& aesParameters = downcast<CryptoAlgorithmAesKeyParams>(parameters);

    if (usagesAreInvalidForCryptoAlgorithmAESCTR(usages)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    auto result = CryptoKeyAES::generate(CryptoAlgorithmIdentifier::AES_CTR, aesParameters.length, extractable, usages);
    if (!result) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    callback(WTFMove(result));
}

void CryptoAlgorithmAESCTR::importKey(CryptoKeyFormat format, KeyData&& data, const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmAESCTRInternal;

    if (usagesAreInvalidForCryptoAlgorithmAESCTR(usages)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    RefPtr<CryptoKeyAES> result;
    switch (format) {
    case CryptoKeyFormat::Raw:
        result = CryptoKeyAES::importRaw(parameters.identifier, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Jwk: {
        auto checkAlgCallback = [](size_t length, const String& alg) -> bool {
            switch (length) {
            case CryptoKeyAES::s_length128:
                return alg.isNull() || alg == ALG128;
            case CryptoKeyAES::s_length192:
                return alg.isNull() || alg == ALG192;
            case CryptoKeyAES::s_length256:
                return alg.isNull() || alg == ALG256;
            }
            return false;
        };
        result = CryptoKeyAES::importJwk(parameters.identifier, WTFMove(std::get<JsonWebKey>(data)), extractable, usages, WTFMove(checkAlgCallback));
        break;
    }
    default:
        exceptionCallback(ExceptionCode::NotSupportedError);
        return;
    }
    if (!result) {
        exceptionCallback(ExceptionCode::DataError);
        return;
    }

    callback(*result);
}

void CryptoAlgorithmAESCTR::exportKey(CryptoKeyFormat format, Ref<CryptoKey>&& key, KeyDataCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmAESCTRInternal;
    const auto& aesKey = downcast<CryptoKeyAES>(key.get());

    if (aesKey.key().isEmpty()) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    KeyData result;
    switch (format) {
    case CryptoKeyFormat::Raw:
        result = Vector<uint8_t>(aesKey.key());
        break;
    case CryptoKeyFormat::Jwk: {
        JsonWebKey jwk = aesKey.exportJwk();
        switch (aesKey.key().size() * 8) {
        case CryptoKeyAES::s_length128:
            jwk.alg = String(ALG128);
            break;
        case CryptoKeyAES::s_length192:
            jwk.alg = String(ALG192);
            break;
        case CryptoKeyAES::s_length256:
            jwk.alg = String(ALG256);
            break;
        default:
            ASSERT_NOT_REACHED();
        }
        result = WTFMove(jwk);
        break;
    }
    default:
        exceptionCallback(ExceptionCode::NotSupportedError);
        return;
    }

    callback(format, WTFMove(result));
}

ExceptionOr<std::optional<size_t>> CryptoAlgorithmAESCTR::getKeyLength(const CryptoAlgorithmParameters& parameters)
{
    return CryptoKeyAES::getKeyLength(parameters);
}

CryptoAlgorithmAESCTR::CounterBlockHelper::CounterBlockHelper(const Vector<uint8_t>& counterVector, size_t counterLength)
    : m_counterLength(counterLength)
{
    using namespace CryptoAlgorithmAESCTRInternal;

    ASSERT(counterVector.size() == counterSize);
    ASSERT(counterLength <= counterSize * 8);
    bool littleEndian = false; // counterVector is stored in big-endian.
    memcpySpan(asMutableByteSpan(m_bits.m_hi), counterVector.span().first(8));
    m_bits.m_hi = flipBytesIfLittleEndian(m_bits.m_hi, littleEndian);
    memcpySpan(asMutableByteSpan(m_bits.m_lo), counterVector.subspan(8));
    m_bits.m_lo = flipBytesIfLittleEndian(m_bits.m_lo, littleEndian);
}

size_t CryptoAlgorithmAESCTR::CounterBlockHelper::countToOverflowSaturating() const
{
    CounterBlockBits counterMask;
    counterMask.set();
    counterMask <<= m_counterLength;
    counterMask = ~counterMask;

    auto countMinusOne = ~m_bits & counterMask;

    CounterBlockBits sizeTypeMask;
    sizeTypeMask.set();
    sizeTypeMask <<= sizeof(size_t) * 8;
    if ((sizeTypeMask & countMinusOne).any()) {
        // Saturating to the size_t max since the count is greater than that.
        return std::numeric_limits<size_t>::max();
    }

    countMinusOne &= ~sizeTypeMask;
    if (countMinusOne.all()) {
        // As all bits are set, adding one would result in an overflow.
        // Return size_t max instead.
        return std::numeric_limits<size_t>::max();
    }

    static_assert(sizeof(size_t) <= sizeof(uint64_t));
    return countMinusOne.m_lo + 1;
}

Vector<uint8_t> CryptoAlgorithmAESCTR::CounterBlockHelper::counterVectorAfterOverflow() const
{
    using namespace CryptoAlgorithmAESCTRInternal;

    CounterBlockBits nonceMask;
    nonceMask.set();
    nonceMask <<= m_counterLength;
    auto bits = m_bits & nonceMask;

    bool littleEndian = false; // counterVector is stored in big-endian.
    Vector<uint8_t> counterVector(counterSize);
    uint64_t hi = flipBytesIfLittleEndian(bits.m_hi, littleEndian);
    memcpySpan(counterVector.mutableSpan(), asByteSpan(hi));
    uint64_t lo = flipBytesIfLittleEndian(bits.m_lo, littleEndian);
    memcpySpan(counterVector.mutableSpan().subspan(8), asByteSpan(lo));

    return counterVector;
}

void CryptoAlgorithmAESCTR::CounterBlockHelper::CounterBlockBits::set()
{
    using namespace CryptoAlgorithmAESCTRInternal;
    m_hi = allBitsSet;
    m_lo = allBitsSet;
}

bool CryptoAlgorithmAESCTR::CounterBlockHelper::CounterBlockBits::all() const
{
    using namespace CryptoAlgorithmAESCTRInternal;
    return m_hi == allBitsSet && m_lo == allBitsSet;
}

bool CryptoAlgorithmAESCTR::CounterBlockHelper::CounterBlockBits::any() const
{
    return m_hi || m_lo;
}

auto CryptoAlgorithmAESCTR::CounterBlockHelper::CounterBlockBits::operator&(const CounterBlockBits& rhs) const -> CounterBlockBits
{
    return { m_hi & rhs.m_hi, m_lo & rhs.m_lo };
}

auto CryptoAlgorithmAESCTR::CounterBlockHelper::CounterBlockBits::operator~() const -> CounterBlockBits
{
    return { ~m_hi, ~m_lo };
}

auto CryptoAlgorithmAESCTR::CounterBlockHelper::CounterBlockBits::operator <<=(unsigned shift) -> CounterBlockBits&
{
    if (shift < 64) {
        m_hi = (m_hi << shift) | m_lo >> (64 - shift);
        m_lo <<= shift;
    } else if (shift < 128) {
        shift -= 64;
        m_hi = m_lo << shift;
        m_lo = 0;
    } else {
        m_hi = 0;
        m_lo = 0;
    }
    return *this;
}

auto CryptoAlgorithmAESCTR::CounterBlockHelper::CounterBlockBits::operator &=(const CounterBlockBits& rhs) -> CounterBlockBits&
{
    m_hi &= rhs.m_hi;
    m_lo &= rhs.m_lo;
    return *this;
}

} // namespace WebCore
