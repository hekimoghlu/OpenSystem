/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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

#include "ContextDestructionObserver.h"
#include "CryptoKeyFormat.h"
#include <JavaScriptCore/Strong.h>
#include <variant>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/WorkQueue.h>

namespace JSC {
class ArrayBufferView;
class ArrayBuffer;
class CallFrame;
}

namespace WebCore {

struct JsonWebKey;

class BufferSource;
class CryptoKey;
class DeferredPromise;

enum class CryptoAlgorithmIdentifier : uint8_t;
enum class CryptoKeyUsage : uint8_t;

class SubtleCrypto : public RefCountedAndCanMakeWeakPtr<SubtleCrypto>, public ContextDestructionObserver {
public:
    static Ref<SubtleCrypto> create(ScriptExecutionContext* context) { return adoptRef(*new SubtleCrypto(context)); }
    ~SubtleCrypto();

    using KeyFormat = CryptoKeyFormat;

    using AlgorithmIdentifier = std::variant<JSC::Strong<JSC::JSObject>, String>;
    using KeyDataVariant = std::variant<RefPtr<JSC::ArrayBufferView>, RefPtr<JSC::ArrayBuffer>, JsonWebKey>;

    void encrypt(JSC::JSGlobalObject&, AlgorithmIdentifier&&, CryptoKey&, BufferSource&& data, Ref<DeferredPromise>&&);
    void decrypt(JSC::JSGlobalObject&, AlgorithmIdentifier&&, CryptoKey&, BufferSource&& data, Ref<DeferredPromise>&&);
    void sign(JSC::JSGlobalObject&, AlgorithmIdentifier&&, CryptoKey&, BufferSource&& data, Ref<DeferredPromise>&&);
    void verify(JSC::JSGlobalObject&, AlgorithmIdentifier&&, CryptoKey&, BufferSource&& signature, BufferSource&& data, Ref<DeferredPromise>&&);
    void digest(JSC::JSGlobalObject&, AlgorithmIdentifier&&, BufferSource&& data, Ref<DeferredPromise>&&);
    void generateKey(JSC::JSGlobalObject&, AlgorithmIdentifier&&, bool extractable, Vector<CryptoKeyUsage>&& keyUsages, Ref<DeferredPromise>&&);
    void deriveKey(JSC::JSGlobalObject&, AlgorithmIdentifier&&, CryptoKey& baseKey, AlgorithmIdentifier&& derivedKeyType, bool extractable, Vector<CryptoKeyUsage>&&, Ref<DeferredPromise>&&);
    void deriveBits(JSC::JSGlobalObject&, AlgorithmIdentifier&&, CryptoKey& baseKey, std::optional<unsigned> length, Ref<DeferredPromise>&&);
    void importKey(JSC::JSGlobalObject&, KeyFormat, KeyDataVariant&&, AlgorithmIdentifier&&, bool extractable, Vector<CryptoKeyUsage>&&, Ref<DeferredPromise>&&);
    void exportKey(KeyFormat, CryptoKey&, Ref<DeferredPromise>&&);
    void wrapKey(JSC::JSGlobalObject&, KeyFormat, CryptoKey&, CryptoKey& wrappingKey, AlgorithmIdentifier&& wrapAlgorithm, Ref<DeferredPromise>&&);
    void unwrapKey(JSC::JSGlobalObject&, KeyFormat, BufferSource&& wrappedKey, CryptoKey& unwrappingKey, AlgorithmIdentifier&& unwrapAlgorithm, AlgorithmIdentifier&& unwrappedKeyAlgorithm, bool extractable, Vector<CryptoKeyUsage>&&, Ref<DeferredPromise>&&);

private:
    explicit SubtleCrypto(ScriptExecutionContext*);

    void addAuthenticatedEncryptionWarningIfNecessary(CryptoAlgorithmIdentifier);
    inline friend RefPtr<DeferredPromise> getPromise(DeferredPromise*, WeakPtr<SubtleCrypto>);

    Ref<WorkQueue> m_workQueue;
    HashMap<DeferredPromise*, Ref<DeferredPromise>> m_pendingPromises;
};

} // namespace WebCore
