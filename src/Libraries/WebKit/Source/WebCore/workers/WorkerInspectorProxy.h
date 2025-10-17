/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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

#include "PageIdentifier.h"
#include "ScriptExecutionContextIdentifier.h"
#include <variant>
#include <wtf/CheckedPtr.h>
#include <wtf/CheckedRef.h>
#include <wtf/FastMalloc.h>
#include <wtf/Function.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

// All of these methods should be called on the Main Thread.
// Used to send messages to the WorkerInspector on the WorkerThread.

namespace WebCore {

class ScriptExecutionContext;
class WorkerThread;

enum class WorkerThreadStartMode;

class WorkerInspectorProxy : public RefCounted<WorkerInspectorProxy>, public CanMakeWeakPtr<WorkerInspectorProxy, WeakPtrFactoryInitialization::Eager> {
    WTF_MAKE_TZONE_ALLOCATED(WorkerInspectorProxy);
    WTF_MAKE_NONCOPYABLE(WorkerInspectorProxy);
public:
    static Ref<WorkerInspectorProxy> create(const String& identifier)
    {
        return adoptRef(*new WorkerInspectorProxy(identifier));
    }

    ~WorkerInspectorProxy();

    // A Worker's inspector messages come in and go out through the Page's WorkerAgent.
    class PageChannel : public CanMakeThreadSafeCheckedPtr<PageChannel> {
        WTF_MAKE_TZONE_ALLOCATED(PageChannel);
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PageChannel);

    public:
        virtual ~PageChannel() = default;

        virtual void ref() const = 0;
        virtual void deref() const = 0;
        virtual void sendMessageFromWorkerToFrontend(WorkerInspectorProxy&, String&&) = 0;
    };

    static Vector<Ref<WorkerInspectorProxy>> proxiesForPage(PageIdentifier);
    static Vector<Ref<WorkerInspectorProxy>> proxiesForWorkerGlobalScope(ScriptExecutionContextIdentifier);

    const URL& url() const { return m_url; }
    const String& name() const { return m_name; }
    const String& identifier() const { return m_identifier; }
    ScriptExecutionContext* scriptExecutionContext() const { return m_scriptExecutionContext.get(); }

    WorkerThreadStartMode workerStartMode(ScriptExecutionContext&);
    void workerStarted(ScriptExecutionContext&, WorkerThread*, const URL&, const String& name);
    void workerTerminated();

    void resumeWorkerIfPaused();
    void connectToWorkerInspectorController(PageChannel&);
    void disconnectFromWorkerInspectorController();
    void sendMessageToWorkerInspectorController(const String&);
    void sendMessageFromWorkerToFrontend(String&&);

private:
    explicit WorkerInspectorProxy(const String& identifier);

    using PageOrWorkerGlobalScopeIdentifier = std::variant<PageIdentifier, ScriptExecutionContextIdentifier>;
    static std::optional<PageOrWorkerGlobalScopeIdentifier> pageOrWorkerGlobalScopeIdentifier(ScriptExecutionContext&);

    void addToProxyMap();
    void removeFromProxyMap();

    RefPtr<ScriptExecutionContext> m_scriptExecutionContext;
    std::optional<PageOrWorkerGlobalScopeIdentifier> m_contextIdentifier;
    RefPtr<WorkerThread> m_workerThread;
    String m_identifier;
    URL m_url;
    String m_name;
    CheckedPtr<PageChannel> m_pageChannel;
};

} // namespace WebCore
