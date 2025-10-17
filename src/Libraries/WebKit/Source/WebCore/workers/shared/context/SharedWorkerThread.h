/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#include "WorkerThread.h"

namespace WebCore {

class WorkerObjectProxy;

class SharedWorkerThread : public WorkerThread {
public:
    template<typename... Args> static Ref<SharedWorkerThread> create(Args&&... args) { return adoptRef(*new SharedWorkerThread(std::forward<Args>(args)...)); }

    SharedWorkerIdentifier identifier() const { return m_identifier; }

private:
    SharedWorkerThread(SharedWorkerIdentifier, const WorkerParameters&, const ScriptBuffer& sourceCode, WorkerLoaderProxy&, WorkerDebuggerProxy&, WorkerObjectProxy&, WorkerBadgeProxy&, WorkerThreadStartMode, const SecurityOrigin& topOrigin, IDBClient::IDBConnectionProxy*, SocketProvider*, JSC::RuntimeFlags);

    Ref<WorkerGlobalScope> createWorkerGlobalScope(const WorkerParameters&, Ref<SecurityOrigin>&&, Ref<SecurityOrigin>&& topOrigin) final;
    ASCIILiteral threadName() const final { return "WebCore: SharedWorker"_s; }

    SharedWorkerIdentifier m_identifier;
    String m_name;
};

} // namespace WebCore
