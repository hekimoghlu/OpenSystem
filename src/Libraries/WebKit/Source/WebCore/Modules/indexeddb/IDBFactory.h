/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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

#include "ExceptionOr.h"
#include "JSDOMPromiseDeferredForward.h"
#include <wtf/Function.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
class JSValue;
}

namespace WebCore {

class IDBOpenDBRequest;
class ScriptExecutionContext;
class SecurityOrigin;

namespace IDBClient {
class IDBConnectionProxy;
}

class IDBFactory : public ThreadSafeRefCounted<IDBFactory> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBFactory);
public:
    static Ref<IDBFactory> create(IDBClient::IDBConnectionProxy&);
    ~IDBFactory();

    struct DatabaseInfo {
        String name;
        uint64_t version;
    };

    ExceptionOr<Ref<IDBOpenDBRequest>> open(ScriptExecutionContext&, const String& name, std::optional<uint64_t> version);
    ExceptionOr<Ref<IDBOpenDBRequest>> deleteDatabase(ScriptExecutionContext&, const String& name);

    ExceptionOr<short> cmp(JSC::JSGlobalObject&, JSC::JSValue first, JSC::JSValue second);

    using IDBDatabasesResponsePromise = DOMPromiseDeferred<IDLSequence<IDLDictionary<IDBFactory::DatabaseInfo>>>;

    void databases(ScriptExecutionContext&, IDBDatabasesResponsePromise&&);

    WEBCORE_EXPORT void getAllDatabaseNames(ScriptExecutionContext&, Function<void(const Vector<String>&)>&&);

private:
    explicit IDBFactory(IDBClient::IDBConnectionProxy&);

    ExceptionOr<Ref<IDBOpenDBRequest>> openInternal(ScriptExecutionContext&, const String& name, uint64_t version);

    Ref<IDBClient::IDBConnectionProxy> m_connectionProxy;
};

} // namespace WebCore
