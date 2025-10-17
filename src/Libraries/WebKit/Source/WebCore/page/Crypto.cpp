/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#include "Crypto.h"

#include "Document.h"
#include "SubtleCrypto.h"
#include <JavaScriptCore/ArrayBufferView.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/UUID.h>

#if OS(DARWIN)
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonRandom.h>
#endif

namespace WebCore {

Crypto::Crypto(ScriptExecutionContext* context)
    : ContextDestructionObserver(context)
    , m_subtle(SubtleCrypto::create(context))
{
}

Crypto::~Crypto() = default;

ExceptionOr<void> Crypto::getRandomValues(ArrayBufferView& array)
{
    if (!isInt(array.getType()) && !isBigInt(array.getType()))
        return Exception { ExceptionCode::TypeMismatchError };
    if (array.byteLength() > 65536)
        return Exception { ExceptionCode::QuotaExceededError };
#if OS(DARWIN)
    auto rc = CCRandomGenerateBytes(array.baseAddress(), array.byteLength());
    RELEASE_ASSERT(rc == kCCSuccess);
#else
    cryptographicallyRandomValues(array.mutableSpan());
#endif
    return { };
}

String Crypto::randomUUID() const
{
    return createVersion4UUIDString();
}

SubtleCrypto& Crypto::subtle()
{
    return m_subtle;
}

} // namespace WebCore
