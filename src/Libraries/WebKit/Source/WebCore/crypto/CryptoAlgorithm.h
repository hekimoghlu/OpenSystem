/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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

#include "CryptoAlgorithmIdentifier.h"
#include "CryptoKey.h"
#include "CryptoKeyFormat.h"
#include "CryptoKeyPair.h"
#include "CryptoKeyUsage.h"
#include "ExceptionOr.h"
#include "JsonWebKey.h"
#include <pal/crypto/CryptoDigest.h>
#include <variant>
#include <wtf/Function.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class CryptoAlgorithmParameters;
class ScriptExecutionContext;

using KeyData = std::variant<Vector<uint8_t>, JsonWebKey>;
using KeyOrKeyPair = std::variant<RefPtr<CryptoKey>, CryptoKeyPair>;

class CryptoAlgorithm : public ThreadSafeRefCounted<CryptoAlgorithm> {
public:
    virtual ~CryptoAlgorithm() = default;

    virtual CryptoAlgorithmIdentifier identifier() const = 0;

    using BoolCallback = Function<void(bool)>;
    using KeyCallback = Function<void(CryptoKey&)>;
    using KeyOrKeyPairCallback = Function<void(KeyOrKeyPair&&)>;
    // FIXME: https://bugs.webkit.org/show_bug.cgi?id=169395
    using VectorCallback = Function<void(const Vector<uint8_t>&)>;
    using VoidCallback = Function<void()>;
    using ExceptionCallback = Function<void(ExceptionCode)>;
    using KeyDataCallback = Function<void(CryptoKeyFormat, KeyData&&)>;

    virtual void encrypt(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&);
    virtual void decrypt(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&);
    virtual void sign(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&);
    virtual void verify(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, Vector<uint8_t>&& signature, Vector<uint8_t>&&, BoolCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&);
    virtual void digest(Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&);
    virtual void generateKey(const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyOrKeyPairCallback&&, ExceptionCallback&&, ScriptExecutionContext&);
    virtual void deriveBits(const CryptoAlgorithmParameters&, Ref<CryptoKey>&&, std::optional<size_t> length, VectorCallback&&, ExceptionCallback&&, ScriptExecutionContext&, WorkQueue&);
    // FIXME: https://bugs.webkit.org/show_bug.cgi?id=169262
    virtual void importKey(CryptoKeyFormat, KeyData&&, const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap, KeyCallback&&, ExceptionCallback&&);
    virtual void exportKey(CryptoKeyFormat, Ref<CryptoKey>&&, KeyDataCallback&&, ExceptionCallback&&);
    virtual void wrapKey(Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&);
    virtual void unwrapKey(Ref<CryptoKey>&&, Vector<uint8_t>&&, VectorCallback&&, ExceptionCallback&&);
    virtual ExceptionOr<std::optional<size_t>> getKeyLength(const CryptoAlgorithmParameters&);

    static void dispatchOperationInWorkQueue(WorkQueue&, ScriptExecutionContext&, VectorCallback&&, ExceptionCallback&&, Function<ExceptionOr<Vector<uint8_t>>()>&&);
    static void dispatchOperationInWorkQueue(WorkQueue&, ScriptExecutionContext&, BoolCallback&&, ExceptionCallback&&, Function<ExceptionOr<bool>()>&&);
    static void dispatchDigest(WorkQueue&, ScriptExecutionContext&, VectorCallback&&, ExceptionCallback&&, Vector<uint8_t>&& message, PAL::CryptoDigest::Algorithm);
};

} // namespace WebCore
