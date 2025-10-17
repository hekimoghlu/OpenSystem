/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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

#if ENABLE(WEB_AUDIO)
#include "WorkerOrWorkletThread.h"
#include "WorkletParameters.h"

namespace WebCore {

class AudioWorkletGlobalScope;
class AudioWorkletMessagingProxy;

class AudioWorkletThread : public WorkerOrWorkletThread {
public:
    static Ref<AudioWorkletThread> create(AudioWorkletMessagingProxy& messagingProxy, WorkletParameters&& parameters)
    {
        return adoptRef(*new AudioWorkletThread(messagingProxy, WTFMove(parameters)));
    }
    ~AudioWorkletThread();

    AudioWorkletGlobalScope* globalScope() const;

    void clearProxies() final;

    // WorkerOrWorkletThread.
    WorkerLoaderProxy* workerLoaderProxy() final;
    WorkerDebuggerProxy* workerDebuggerProxy() const final;

    AudioWorkletMessagingProxy* messagingProxy() { return m_messagingProxy.get(); }

private:
    AudioWorkletThread(AudioWorkletMessagingProxy&, WorkletParameters&&);

    // WorkerOrWorkletThread.
    Ref<Thread> createThread() final;
    RefPtr<WorkerOrWorkletGlobalScope> createGlobalScope() final;

    CheckedPtr<AudioWorkletMessagingProxy> m_messagingProxy;
    WorkletParameters m_parameters;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
