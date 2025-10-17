/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#include "CryptoAlgorithmPBKDF2.h"

#include "CryptoAlgorithmPbkdf2Params.h"
#include "CryptoKeyRaw.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

Ref<CryptoAlgorithm> CryptoAlgorithmPBKDF2::create()
{
    return adoptRef(*new CryptoAlgorithmPBKDF2);
}

CryptoAlgorithmIdentifier CryptoAlgorithmPBKDF2::identifier() const
{
    return s_identifier;
}

void CryptoAlgorithmPBKDF2::deriveBits(const CryptoAlgorithmParameters& parameters, Ref<CryptoKey>&& baseKey, std::optional<size_t> length, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    if (!length || *length % 8) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [parameters = crossThreadCopy(downcast<CryptoAlgorithmPbkdf2Params>(parameters)), baseKey = WTFMove(baseKey), length] {
            return *length ? platformDeriveBits(parameters, downcast<CryptoKeyRaw>(baseKey.get()), *length) : Vector<uint8_t>();
        });
}

void CryptoAlgorithmPBKDF2::importKey(CryptoKeyFormat format, KeyData&& data, const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    if (format != CryptoKeyFormat::Raw) {
        exceptionCallback(ExceptionCode::NotSupportedError);
        return;
    }
    if (usages & (CryptoKeyUsageEncrypt | CryptoKeyUsageDecrypt | CryptoKeyUsageSign | CryptoKeyUsageVerify | CryptoKeyUsageWrapKey | CryptoKeyUsageUnwrapKey)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }
    if (extractable) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    callback(CryptoKeyRaw::create(parameters.identifier, WTFMove(std::get<Vector<uint8_t>>(data)), usages));
}

ExceptionOr<std::optional<size_t>> CryptoAlgorithmPBKDF2::getKeyLength(const CryptoAlgorithmParameters&)
{
    return std::optional<size_t>();
}

} // namespace WebCore
