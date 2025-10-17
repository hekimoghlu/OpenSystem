/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
#include "CryptoAlgorithm.h"

#include "ScriptExecutionContext.h"

namespace WebCore {

void CryptoAlgorithm::encrypt(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&, WorkQueue&)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::decrypt(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&, WorkQueue&)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::sign(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&, WorkQueue&)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::verify(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, Vector<uint8_t>&&, BoolCallback&&, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&, WorkQueue&)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::digest(Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&, WorkQueue&)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::generateKey(const CryptoAlgorithmParameters&, bool, CryptoKeyUsageBitmap, KeyOrKeyPairCallback&&, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::deriveBits(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, std::optional<size_t>, VectorCallback&&, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&, WorkQueue&)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::importKey(CryptoKeyFormat, KeyData&&, const CryptoAlgorithmParameters&, bool, CryptoKeyUsageBitmap, KeyCallback&&, ExceptionCallback&& exceptionCallback)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::exportKey(CryptoKeyFormat, Ref<CryptoKey>&&, KeyDataCallback&&, ExceptionCallback&& exceptionCallback)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::wrapKey(Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&& exceptionCallback)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

void CryptoAlgorithm::unwrapKey(Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&& exceptionCallback)
{
    exceptionCallback(ExceptionCode::NotSupportedError);
}

ExceptionOr<std::optional<size_t>> CryptoAlgorithm::getKeyLength(const CryptoAlgorithmParameters&)
{
    return Exception { ExceptionCode::NotSupportedError };
}

template<typename ResultCallbackType, typename OperationType>
static void dispatchAlgorithmOperation(WorkQueue& workQueue, ScriptExecutionContext& context, ResultCallbackType&& callback, CryptoAlgorithm::ExceptionCallback&& exceptionCallback, OperationType&& operation)
{
    workQueue.dispatch(
        [operation = std::forward<OperationType>(operation), callback = std::forward<ResultCallbackType>(callback), exceptionCallback = WTFMove(exceptionCallback), contextIdentifier = context.identifier()]() mutable {
            auto result = operation();
            ScriptExecutionContext::postTaskTo(contextIdentifier, [result = crossThreadCopy(WTFMove(result)), callback = WTFMove(callback), exceptionCallback = WTFMove(exceptionCallback)](auto&) mutable {
                if (result.hasException()) {
                    exceptionCallback(result.releaseException().code());
                    return;
                }
                callback(result.releaseReturnValue());
            });
        });
}

void CryptoAlgorithm::dispatchOperationInWorkQueue(WorkQueue& workQueue, ScriptExecutionContext& context, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, Function<ExceptionOr<Vector<uint8_t>>()>&& operation)
{
    dispatchAlgorithmOperation(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback), WTFMove(operation));
}

void CryptoAlgorithm::dispatchOperationInWorkQueue(WorkQueue& workQueue, ScriptExecutionContext& context, BoolCallback&& callback, ExceptionCallback&& exceptionCallback, Function<ExceptionOr<bool>()>&& operation)
{
    dispatchAlgorithmOperation(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback), WTFMove(operation));
}

void CryptoAlgorithm::dispatchDigest(WorkQueue& workQueue, ScriptExecutionContext& context, VectorCallback&& callback, ExceptionCallback&&exceptionCallback, Vector<uint8_t>&& message, PAL::CryptoDigest::Algorithm algo)
{
    workQueue.dispatch([message = WTFMove(message), callback = WTFMove(callback), contextIdentifier = context.identifier(), exceptionCallback = WTFMove(exceptionCallback), algo]() mutable {
        auto digest = PAL::CryptoDigest::create(algo);
        if (!digest) {
            exceptionCallback(ExceptionCode::OperationError);
            return;
        }
        digest->addBytes(message.span());
        auto result = digest->computeHash();

        ScriptExecutionContext::postTaskTo(contextIdentifier, [callback = WTFMove(callback), result = WTFMove(result), exceptionCallback = WTFMove(exceptionCallback)](auto&) mutable {
            callback(WTFMove(result));
        });
    });
}

} // namespace WebCore
