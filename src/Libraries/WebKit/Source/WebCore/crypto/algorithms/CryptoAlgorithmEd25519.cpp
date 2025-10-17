/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
#include "CryptoAlgorithmEd25519.h"

#include "CryptoKeyOKP.h"
#include "NotImplemented.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

#if !PLATFORM(COCOA) && !USE(GCRYPT)
ExceptionOr<Vector<uint8_t>> CryptoAlgorithmEd25519::platformSign(const CryptoKeyOKP&, const Vector<uint8_t>&)
{
    ASSERT_NOT_REACHED();
    notImplemented();
    return Exception { ExceptionCode::NotSupportedError };
}

ExceptionOr<bool> CryptoAlgorithmEd25519::platformVerify(const CryptoKeyOKP&, const Vector<uint8_t>&, const Vector<uint8_t>&)
{
    ASSERT_NOT_REACHED();
    notImplemented();
    return Exception { ExceptionCode::NotSupportedError };
}
#endif // PLATFORM(COCOA)

Ref<CryptoAlgorithm> CryptoAlgorithmEd25519::create()
{
    return adoptRef(*new CryptoAlgorithmEd25519);
}

void CryptoAlgorithmEd25519::generateKey(const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap usages, KeyOrKeyPairCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&)
{
    if (usages & (CryptoKeyUsageEncrypt | CryptoKeyUsageDecrypt | CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits | CryptoKeyUsageWrapKey | CryptoKeyUsageUnwrapKey)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    auto result = CryptoKeyOKP::generatePair(CryptoAlgorithmIdentifier::Ed25519, CryptoKeyOKP::NamedCurve::Ed25519, extractable, usages);
    if (result.hasException()) {
        exceptionCallback(result.releaseException().code());
        return;
    }

    auto pair = result.releaseReturnValue();
    pair.publicKey->setUsagesBitmap(pair.publicKey->usagesBitmap() & CryptoKeyUsageVerify);
    pair.privateKey->setUsagesBitmap(pair.privateKey->usagesBitmap() & CryptoKeyUsageSign);
    callback(WTFMove(pair));
}

void CryptoAlgorithmEd25519::sign(const CryptoAlgorithmParameters&, Ref<CryptoKey>&& key, Vector<uint8_t>&& data, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    if (key->type() != CryptoKeyType::Private) {
        exceptionCallback(ExceptionCode::InvalidAccessError);
        return;
    }
    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [key = WTFMove(key), data = WTFMove(data)] {
            return platformSign(downcast<CryptoKeyOKP>(key.get()), data);
    });
}

void CryptoAlgorithmEd25519::verify(const CryptoAlgorithmParameters&, Ref<CryptoKey>&& key, Vector<uint8_t>&& signature, Vector<uint8_t>&& data, BoolCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    if (key->type() != CryptoKeyType::Public) {
        exceptionCallback(ExceptionCode::InvalidAccessError);
        return;
    }
    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [key = WTFMove(key), signature = WTFMove(signature), data = WTFMove(data)] {
            return platformVerify(downcast<CryptoKeyOKP>(key.get()), signature, data);
        });
}

void CryptoAlgorithmEd25519::importKey(CryptoKeyFormat format, KeyData&& data, const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap usages, KeyCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    RefPtr<CryptoKeyOKP> result;
    switch (format) {
    case CryptoKeyFormat::Jwk: {
        JsonWebKey key = WTFMove(std::get<JsonWebKey>(data));
        if (usages && ((!key.d.isNull() && (usages ^ CryptoKeyUsageSign)) || (key.d.isNull() && (usages ^ CryptoKeyUsageVerify)))) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        if (usages && !key.use.isNull() && key.use != "sig"_s) {
            exceptionCallback(ExceptionCode::DataError);
            return;
        }
        result = CryptoKeyOKP::importJwk(CryptoAlgorithmIdentifier::Ed25519, CryptoKeyOKP::NamedCurve::Ed25519, WTFMove(key), extractable, usages);
        break;
    }
    case CryptoKeyFormat::Raw:
        if (usages && (usages ^ CryptoKeyUsageVerify)) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyOKP::importRaw(CryptoAlgorithmIdentifier::Ed25519, CryptoKeyOKP::NamedCurve::Ed25519, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Spki:
        if (usages && (usages ^ CryptoKeyUsageVerify)) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyOKP::importSpki(CryptoAlgorithmIdentifier::Ed25519, CryptoKeyOKP::NamedCurve::Ed25519, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Pkcs8:
        if (usages && (usages ^ CryptoKeyUsageSign)) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyOKP::importPkcs8(CryptoAlgorithmIdentifier::Ed25519, CryptoKeyOKP::NamedCurve::Ed25519, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    }
    if (!result) {
        exceptionCallback(ExceptionCode::DataError);
        return;
    }
    callback(*result);
}

void CryptoAlgorithmEd25519::exportKey(CryptoKeyFormat format, Ref<CryptoKey>&& key, KeyDataCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    const auto& okpKey = downcast<CryptoKeyOKP>(key.get());
    if (!okpKey.keySizeInBits()) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }
    KeyData result;
    switch (format) {
    case CryptoKeyFormat::Jwk: {
        auto jwk = okpKey.exportJwk();
        if (jwk.hasException()) {
            exceptionCallback(jwk.releaseException().code());
            return;
        }
        result = jwk.releaseReturnValue();
        break;
    }
    case CryptoKeyFormat::Raw: {
        auto raw = okpKey.exportRaw();
        if (raw.hasException()) {
            exceptionCallback(raw.releaseException().code());
            return;
        }
        result = raw.releaseReturnValue();
        break;
    }
    case CryptoKeyFormat::Spki: {
        auto spki = okpKey.exportSpki();
        if (spki.hasException()) {
            exceptionCallback(spki.releaseException().code());
            return;
        }
        result = spki.releaseReturnValue();
        break;
    }
    case CryptoKeyFormat::Pkcs8: {
        auto pkcs8 = okpKey.exportPkcs8();
        if (pkcs8.hasException()) {
            exceptionCallback(pkcs8.releaseException().code());
            return;
        }
        result = pkcs8.releaseReturnValue();
        break;
    }
    }
    callback(format, WTFMove(result));
}

} // namespace WebCore
