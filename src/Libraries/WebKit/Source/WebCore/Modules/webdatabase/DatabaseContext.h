/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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

#include "ActiveDOMObject.h"
#include "Document.h"
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>

#if PLATFORM(IOS_FAMILY)
#include <wtf/Threading.h>
#endif

namespace WebCore {

class Database;
class DatabaseDetails;
class DatabaseTaskSynchronizer;
class DatabaseThread;
class SecurityOrigin;
class SecurityOriginData;

class DatabaseContext final : public ThreadSafeRefCounted<DatabaseContext>, private ActiveDOMObject {
public:
    // ActiveDOMObject.
    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    virtual ~DatabaseContext();

    DatabaseThread* existingDatabaseThread() const { return m_databaseThread.get(); }
    DatabaseThread* databaseThread();

    void setHasOpenDatabases() { m_hasOpenDatabases = true; }
    bool hasOpenDatabases() const { return m_hasOpenDatabases; }

    // When the database cleanup is done, the sychronizer will be signalled.
    bool stopDatabases(DatabaseTaskSynchronizer*);

    bool allowDatabaseAccess() const;
    void databaseExceededQuota(const String& name, DatabaseDetails);

    Document* document() const { return downcast<Document>(ActiveDOMObject::scriptExecutionContext()); }
    const SecurityOriginData& securityOrigin() const;

    bool isContextThread() const;

private:
    explicit DatabaseContext(Document&);

    void stopDatabases() { stopDatabases(nullptr); }

    void contextDestroyed() override;

    // ActiveDOMObject.
    void stop() override;

    RefPtr<DatabaseThread> m_databaseThread;
    bool m_hasOpenDatabases { false }; // This never changes back to false, even after the database thread is closed.
    bool m_hasRequestedTermination { false };

    friend class DatabaseManager;
};

} // namespace WebCore
