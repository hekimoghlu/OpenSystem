/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

#include "SharedWorkerIdentifier.h"
#include "TransferredMessagePort.h"
#include <wtf/AbstractRefCounted.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ScriptExecutionContext;
class SharedWorkerThreadProxy;

class SharedWorkerContextManager {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SharedWorkerContextManager, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static SharedWorkerContextManager& singleton();

    SharedWorkerThreadProxy* sharedWorker(SharedWorkerIdentifier) const;
    void stopSharedWorker(SharedWorkerIdentifier);
    void suspendSharedWorker(SharedWorkerIdentifier);
    void resumeSharedWorker(SharedWorkerIdentifier);
    WEBCORE_EXPORT void stopAllSharedWorkers();

    class Connection : public AbstractRefCounted {
        WTF_MAKE_TZONE_ALLOCATED_EXPORT(Connection, WEBCORE_EXPORT);
    public:
        virtual ~Connection() { }
        virtual void establishConnection(CompletionHandler<void()>&&) = 0;
        virtual void postErrorToWorkerObject(SharedWorkerIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrrorEvent) = 0;
        virtual void sharedWorkerTerminated(SharedWorkerIdentifier) = 0;
        bool isClosed() const { return m_isClosed; }

    protected:
        void setAsClosed() { m_isClosed = true; }

        // IPC message handlers.
        WEBCORE_EXPORT void postConnectEvent(SharedWorkerIdentifier, TransferredMessagePort&&, String&& sourceOrigin, CompletionHandler<void(bool)>&&);
        WEBCORE_EXPORT void terminateSharedWorker(SharedWorkerIdentifier);
        WEBCORE_EXPORT void suspendSharedWorker(SharedWorkerIdentifier);
        WEBCORE_EXPORT void resumeSharedWorker(SharedWorkerIdentifier);

    private:
        bool m_isClosed { false };
    };

    WEBCORE_EXPORT void setConnection(RefPtr<Connection>&&);
    WEBCORE_EXPORT Connection* connection() const;

    WEBCORE_EXPORT void registerSharedWorkerThread(Ref<SharedWorkerThreadProxy>&&);

    void forEachSharedWorker(const Function<Function<void(ScriptExecutionContext&)>()>&);

private:
    friend class NeverDestroyed<SharedWorkerContextManager>;

    SharedWorkerContextManager() = default;

    RefPtr<Connection> m_connection;
    HashMap<SharedWorkerIdentifier, Ref<SharedWorkerThreadProxy>> m_workerMap;
};

} // namespace WebCore
