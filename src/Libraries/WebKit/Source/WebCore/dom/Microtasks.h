/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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

#include "EventLoop.h"
#include <JavaScriptCore/MicrotaskQueue.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class VM;
} // namespace JSC

namespace WebCore {

class WebCoreMicrotaskDispatcher : public JSC::MicrotaskDispatcher {
    WTF_MAKE_TZONE_ALLOCATED(WebCoreMicrotaskDispatcher);
public:
    WebCoreMicrotaskDispatcher(Type type, EventLoopTaskGroup& group)
        : JSC::MicrotaskDispatcher(type)
        , m_group(group)
    {
    }

    bool isRunnable() const final
    {
        return currentRunnability() == JSC::QueuedTask::Result::Executed;
    }

    JSC::QueuedTask::Result currentRunnability() const;

private:
    WeakPtr<EventLoopTaskGroup> m_group;
};

class MicrotaskQueue final {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(MicrotaskQueue, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT MicrotaskQueue(JSC::VM&, EventLoop&);
    WEBCORE_EXPORT ~MicrotaskQueue();

    WEBCORE_EXPORT void append(JSC::QueuedTask&&);
    WEBCORE_EXPORT void performMicrotaskCheckpoint();

    WEBCORE_EXPORT void addCheckpointTask(std::unique_ptr<EventLoopTask>&&);

    bool isEmpty() const { return m_microtaskQueue.isEmpty(); }
    bool hasMicrotasksForFullyActiveDocument() const;

    static void runJSMicrotask(JSC::JSGlobalObject*, JSC::VM&, JSC::QueuedTask&);

private:
    JSC::VM& vm() const { return m_vm.get(); }

    bool m_performingMicrotaskCheckpoint { false };
    // For the main thread the VM lives forever. For workers it's lifetime is tied to our owning WorkerGlobalScope. Regardless, we retain the VM here to be safe.
    Ref<JSC::VM> m_vm;
    WeakPtr<EventLoop> m_eventLoop;
    JSC::MicrotaskQueue m_microtaskQueue;

    EventLoop::TaskVector m_checkpointTasks;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WebCoreMicrotaskDispatcher)
    static bool isType(const JSC::MicrotaskDispatcher& dispatcher) { return dispatcher.isWebCoreMicrotaskDispatcher(); }
SPECIALIZE_TYPE_TRAITS_END()
