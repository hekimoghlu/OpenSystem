/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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

#include "TransferredMessagePort.h"
#include "WorkerGlobalScope.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SharedWorkerThread;
struct WorkerParameters;

class SharedWorkerGlobalScope final : public WorkerGlobalScope {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SharedWorkerGlobalScope);
public:
    template<typename... Args> static Ref<SharedWorkerGlobalScope> create(Args&&... args)
    {
        auto scope = adoptRef(*new SharedWorkerGlobalScope(std::forward<Args>(args)...));
        scope->addToContextsMap();
        return scope;
    }
    ~SharedWorkerGlobalScope();

    Type type() const final { return Type::SharedWorker; }
    const String& name() const { return m_name; }
    SharedWorkerThread& thread();

    void postConnectEvent(TransferredMessagePort&&, const String& sourceOrigin);

private:
    SharedWorkerGlobalScope(const String& name, const WorkerParameters&, Ref<SecurityOrigin>&&, SharedWorkerThread&, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy*, SocketProvider*, std::unique_ptr<WorkerClient>&&);

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::SharedWorkerGlobalScope; }
    FetchOptions::Destination destination() const final { return FetchOptions::Destination::Sharedworker; }

    String m_name;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SharedWorkerGlobalScope)
static bool isType(const WebCore::ScriptExecutionContext& context)
{
    auto* global = dynamicDowncast<WebCore::WorkerGlobalScope>(context);
    return global && global->type() == WebCore::WorkerGlobalScope::Type::SharedWorker;
}
static bool isType(const WebCore::WorkerGlobalScope& context) { return context.type() == WebCore::WorkerGlobalScope::Type::SharedWorker; }
SPECIALIZE_TYPE_TRAITS_END()
