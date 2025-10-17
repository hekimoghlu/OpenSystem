/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "CryptoAlgorithmAESKW.h"

#include "CryptoKeyAES.h"
#include <CommonCrypto/CommonCrypto.h>
#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwift.h>
#endif

namespace WebCore {

#if !HAVE(SWIFT_CPP_INTEROP)
static ExceptionOr<Vector<uint8_t>> wrapKeyAESKW(const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    Vector<uint8_t> result(CCSymmetricWrappedSize(kCCWRAPAES, data.size()));
    size_t resultSize = result.size();
    if (CCSymmetricKeyWrap(kCCWRAPAES, CCrfc3394_iv, CCrfc3394_ivLen, key.data(), key.size(), data.data(), data.size(), result.data(), &resultSize))
        return Exception { ExceptionCode::OperationError };

    result.shrink(resultSize);
    return WTFMove(result);
}

static ExceptionOr<Vector<uint8_t>> unwrapKeyAESKW(const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    Vector<uint8_t> result(CCSymmetricUnwrappedSize(kCCWRAPAES, data.size()));
    size_t resultSize = result.size();

    if (resultSize % 8)
        return Exception { ExceptionCode::OperationError };

    if (CCSymmetricKeyUnwrap(kCCWRAPAES, CCrfc3394_iv, CCrfc3394_ivLen, key.data(), key.size(), data.data(), data.size(), result.data(), &resultSize))
        return Exception { ExceptionCode::OperationError };

    result.shrink(resultSize);
    return WTFMove(result);
}
#else
static ExceptionOr<Vector<uint8_t>> wrapKeyAESKWCryptoKit(const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    auto rv = PAL::AesKw::wrap(data.span(), key.span());
    if (rv.errorCode != Cpp::ErrorCodes::Success)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(rv.result);
}

static ExceptionOr<Vector<uint8_t>> unwrapKeyAESKWCryptoKit(const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    auto rv = PAL::AesKw::unwrap(data.span(), key.span());
    if (rv.errorCode != Cpp::ErrorCodes::Success)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(rv.result);
}
#endif

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESKW::platformWrapKey(const CryptoKeyAES& key, const Vector<uint8_t>& data)
{
#if HAVE(SWIFT_CPP_INTEROP)
    return wrapKeyAESKWCryptoKit(key.key(), data);
#else
    return wrapKeyAESKW(key.key(), data);
#endif
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESKW::platformUnwrapKey(const CryptoKeyAES& key, const Vector<uint8_t>& data)
{
#if HAVE(SWIFT_CPP_INTEROP)
    return unwrapKeyAESKWCryptoKit(key.key(), data);
#else
    return unwrapKeyAESKW(key.key(), data);
#endif
}

} // namespace WebCore
