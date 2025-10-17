/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

#include "IDBActiveDOMObject.h"
#include "IDBDatabaseNameAndVersion.h"
#include "IDBResourceIdentifier.h"
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class ScriptExecutionContext;

namespace IDBClient {
class IDBConnectionProxy;
}

class WEBCORE_EXPORT IDBDatabaseNameAndVersionRequest final : public ThreadSafeRefCounted<IDBDatabaseNameAndVersionRequest>, public IDBActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(IDBDatabaseNameAndVersionRequest, WEBCORE_EXPORT);
public:
    using InfoCallback = Function<void(std::optional<Vector<IDBDatabaseNameAndVersion>>&&)>;

    static Ref<IDBDatabaseNameAndVersionRequest> create(ScriptExecutionContext&, IDBClient::IDBConnectionProxy&, InfoCallback&&);

    ~IDBDatabaseNameAndVersionRequest();

    const IDBResourceIdentifier& resourceIdentifier() const;

    // ActiveDOMObject.
    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    void complete(std::optional<Vector<IDBDatabaseNameAndVersion>>&&);

private:
    IDBDatabaseNameAndVersionRequest(ScriptExecutionContext&, IDBClient::IDBConnectionProxy&, InfoCallback&&);

    // ActiveDOMObject.
    bool virtualHasPendingActivity() const final;
    void stop() final;

    Ref<IDBClient::IDBConnectionProxy> m_connectionProxy;
    IDBResourceIdentifier m_resourceIdentifier;
    InfoCallback m_callback;
};

inline const IDBResourceIdentifier& IDBDatabaseNameAndVersionRequest::resourceIdentifier() const
{
    return m_resourceIdentifier;
}

inline IDBDatabaseNameAndVersionRequest::~IDBDatabaseNameAndVersionRequest()
{
    ASSERT(canCurrentThreadAccessThreadLocalData(originThread()));
}

} // namespace WebCore
