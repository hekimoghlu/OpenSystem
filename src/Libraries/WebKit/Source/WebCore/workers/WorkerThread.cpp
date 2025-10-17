/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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
#include "WorkerThread.h"

#include "AdvancedPrivacyProtections.h"
#include "IDBConnectionProxy.h"
#include "ScriptSourceCode.h"
#include "SecurityOrigin.h"
#include "SocketProvider.h"
#include "WorkerBadgeProxy.h"
#include "WorkerDebuggerProxy.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerReportingProxy.h"
#include "WorkerScriptFetcher.h"
#include <JavaScriptCore/ScriptCallStack.h>
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Threading.h>

namespace WebCore {

static std::atomic<unsigned> workerThreadCounter { 0 };

unsigned WorkerThread::workerThreadCount()
{
    return workerThreadCounter;
}

WorkerParameters WorkerParameters::isolatedCopy() const
{
    return {
        scriptURL.isolatedCopy(),
        ownerURL.isolatedCopy(),
        name.isolatedCopy(),
        inspectorIdentifier.isolatedCopy(),
        userAgent.isolatedCopy(),
        isOnline,
        contentSecurityPolicyResponseHeaders.isolatedCopy(),
        shouldBypassMainWorldContentSecurityPolicy,
        crossOriginEmbedderPolicy.isolatedCopy(),
        timeOrigin,
        referrerPolicy,
        workerType,
        credentials,
        settingsValues.isolatedCopy(),
        workerThreadMode,
        sessionID,
        crossThreadCopy(serviceWorkerData),
        clientIdentifier,
        advancedPrivacyProtections,
        noiseInjectionHashSalt
    };
}

struct WorkerThreadStartupData {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WorkerThreadStartupData);
    WTF_MAKE_NONCOPYABLE(WorkerThreadStartupData);
public:
    WorkerThreadStartupData(const WorkerParameters& params, const ScriptBuffer& sourceCode, WorkerThreadStartMode, const SecurityOrigin& topOrigin);

    WorkerParameters params;
    Ref<SecurityOrigin> origin;
    ScriptBuffer sourceCode;
    WorkerThreadStartMode startMode;
    Ref<SecurityOrigin> topOrigin;
};

WorkerThreadStartupData::WorkerThreadStartupData(const WorkerParameters& other, const ScriptBuffer& sourceCode, WorkerThreadStartMode startMode, const SecurityOrigin& topOrigin)
    : params(other.isolatedCopy())
    , origin(SecurityOrigin::create(other.scriptURL)->isolatedCopy())
    , sourceCode(sourceCode.isolatedCopy())
    , startMode(startMode)
    , topOrigin(topOrigin.isolatedCopy())
{
}

WorkerThread::WorkerThread(const WorkerParameters& params, const ScriptBuffer& sourceCode, WorkerLoaderProxy& workerLoaderProxy, WorkerDebuggerProxy& workerDebuggerProxy, WorkerReportingProxy& workerReportingProxy, WorkerBadgeProxy& badgeProxy, WorkerThreadStartMode startMode, const SecurityOrigin& topOrigin, IDBClient::IDBConnectionProxy* connectionProxy, SocketProvider* socketProvider, JSC::RuntimeFlags runtimeFlags)
    : WorkerOrWorkletThread(params.inspectorIdentifier.isolatedCopy(), params.workerThreadMode)
    , m_workerLoaderProxy(&workerLoaderProxy)
    , m_workerDebuggerProxy(&workerDebuggerProxy)
    , m_workerReportingProxy(&workerReportingProxy)
    , m_workerBadgeProxy(&badgeProxy)
    , m_runtimeFlags(runtimeFlags)
    , m_startupData(makeUnique<WorkerThreadStartupData>(params, sourceCode, startMode, topOrigin))
    , m_idbConnectionProxy(connectionProxy)
    , m_socketProvider(socketProvider)
{
    ++workerThreadCounter;
}

WorkerThread::~WorkerThread()
{
    ASSERT(workerThreadCounter);
    --workerThreadCounter;
}

WorkerLoaderProxy* WorkerThread::workerLoaderProxy()
{
    return m_workerLoaderProxy.get();
}

WorkerBadgeProxy* WorkerThread::workerBadgeProxy() const
{
    return m_workerBadgeProxy.get();
}

WorkerDebuggerProxy* WorkerThread::workerDebuggerProxy() const
{
    return m_workerDebuggerProxy.get();
}

WorkerReportingProxy* WorkerThread::workerReportingProxy() const
{
    return m_workerReportingProxy.get();
}

Ref<Thread> WorkerThread::createThread()
{
    if (is<WorkerMainRunLoop>(runLoop())) {
        // This worker should run on the main thread.
        RunLoop::main().dispatch([protectedThis = Ref { *this }] {
            protectedThis->workerOrWorkletThread();
        });
        ASSERT(isMainThread());
        return Thread::current();
    }

    return Thread::create(threadName(), [this] {
        workerOrWorkletThread();
    }, ThreadType::JavaScript);
}

RefPtr<WorkerOrWorkletGlobalScope> WorkerThread::createGlobalScope()
{
    return createWorkerGlobalScope(m_startupData->params, WTFMove(m_startupData->origin), WTFMove(m_startupData->topOrigin));
}

bool WorkerThread::shouldWaitForWebInspectorOnStartup() const
{
    return m_startupData->startMode == WorkerThreadStartMode::WaitForInspector;
}

void WorkerThread::evaluateScriptIfNecessary(String& exceptionMessage)
{
    SetForScope isInStaticScriptEvaluation(m_isInStaticScriptEvaluation, true);

    // We are currently holding only the initial script code. If the WorkerType is Module, we should fetch the entire graph before executing the rest of this.
    // We invoke module loader as if we are executing inline module script tag in Document.

    Ref globalScope = *this->globalScope();

    WeakPtr<ScriptBufferSourceProvider> sourceProvider;
    if (m_startupData->params.workerType == WorkerType::Classic) {
        ScriptSourceCode sourceCode(m_startupData->sourceCode, URL(m_startupData->params.scriptURL));
        sourceProvider = static_cast<ScriptBufferSourceProvider&>(sourceCode.provider());
        globalScope->script()->evaluate(sourceCode, &exceptionMessage);
        finishedEvaluatingScript();
    } else {
        auto parameters = ModuleFetchParameters::create(JSC::ScriptFetchParameters::Type::JavaScript, emptyString(), /* isTopLevelModule */ true);
        auto scriptFetcher = WorkerScriptFetcher::create(WTFMove(parameters), globalScope->credentials(), globalScope->destination(), globalScope->referrerPolicy());
        ScriptSourceCode sourceCode(m_startupData->sourceCode, URL(m_startupData->params.scriptURL), { }, { }, JSC::SourceProviderSourceType::Module, scriptFetcher.copyRef());
        sourceProvider = static_cast<ScriptBufferSourceProvider&>(sourceCode.provider());
        bool success = globalScope->script()->loadModuleSynchronously(scriptFetcher.get(), sourceCode);
        if (success) {
            if (auto error = scriptFetcher->error()) {
                if (std::optional<LoadableScript::ConsoleMessage> message = error->consoleMessage)
                    exceptionMessage = message->message;
                else
                    exceptionMessage = "Importing a module script failed."_s;
                globalScope->reportErrorToWorkerObject(exceptionMessage);
            } else if (!scriptFetcher->wasCanceled()) {
                globalScope->script()->linkAndEvaluateModule(scriptFetcher.get(), sourceCode, &exceptionMessage);
                finishedEvaluatingScript();
            }
        }
    }
    if (sourceProvider)
        globalScope->setMainScriptSourceProvider(*sourceProvider);

    // Free the startup data to cause its member variable deref's happen on the worker's thread (since
    // all ref/derefs of these objects are happening on the thread at this point). Note that
    // WorkerThread::~WorkerThread happens on a different thread where it was created.
    m_startupData = nullptr;
}

IDBClient::IDBConnectionProxy* WorkerThread::idbConnectionProxy()
{
    return m_idbConnectionProxy.get();
}

SocketProvider* WorkerThread::socketProvider()
{
    return m_socketProvider.get();
}

WorkerGlobalScope* WorkerThread::globalScope()
{
    ASSERT(!thread() || thread() == &Thread::current());
    return downcast<WorkerGlobalScope>(WorkerOrWorkletThread::globalScope());
}

void WorkerThread::clearProxies()
{
    m_workerLoaderProxy = nullptr;
    m_workerDebuggerProxy = nullptr;
    m_workerReportingProxy = nullptr;
    m_workerBadgeProxy = nullptr;
}

} // namespace WebCore
