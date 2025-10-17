/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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

#include "FetchRequestCredentials.h"
#include "MessageWithMessagePorts.h"
#include <JavaScriptCore/RuntimeFlags.h>
#include <wtf/Function.h>
#include <wtf/MonotonicTime.h>

namespace WebCore {

class ContentSecurityPolicyResponseHeaders;
class ScriptBuffer;
class ScriptExecutionContext;
class Worker;
struct WorkerInitializationData;
enum class ReferrerPolicy : uint8_t;
enum class WorkerType : bool;

// A proxy to talk to the worker context.
class WorkerGlobalScopeProxy {
public:
    static WorkerGlobalScopeProxy& create(Worker&);

    virtual void startWorkerGlobalScope(const URL& scriptURL, PAL::SessionID, const String& name, WorkerInitializationData&&, const ScriptBuffer& sourceCode, const ContentSecurityPolicyResponseHeaders&, bool shouldBypassMainWorldContentSecurityPolicy, const CrossOriginEmbedderPolicy&, MonotonicTime timeOrigin, ReferrerPolicy, WorkerType, FetchRequestCredentials, JSC::RuntimeFlags) = 0;
    virtual void terminateWorkerGlobalScope() = 0;
    virtual void postMessageToWorkerGlobalScope(MessageWithMessagePorts&&) = 0;
    virtual void postTaskToWorkerGlobalScope(Function<void(ScriptExecutionContext&)>&&) = 0;
    virtual bool askedToTerminate() const = 0;
    virtual void workerObjectDestroyed() = 0;
    virtual void notifyNetworkStateChange(bool isOnline) = 0;

    virtual void suspendForBackForwardCache() = 0;
    virtual void resumeForBackForwardCache() = 0;

protected:
    virtual ~WorkerGlobalScopeProxy() = default;
};

} // namespace WebCore
