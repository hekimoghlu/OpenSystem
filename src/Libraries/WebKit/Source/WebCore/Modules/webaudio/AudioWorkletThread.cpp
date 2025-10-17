/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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

#if ENABLE(WEB_AUDIO)
#include "AudioWorkletThread.h"

#include "AudioWorkletGlobalScope.h"
#include "AudioWorkletMessagingProxy.h"
#include <wtf/Threading.h>

namespace WebCore {

AudioWorkletThread::AudioWorkletThread(AudioWorkletMessagingProxy& messagingProxy, WorkletParameters&& parameters)
    : WorkerOrWorkletThread(parameters.identifier.isolatedCopy())
    , m_messagingProxy(&messagingProxy)
    , m_parameters(WTFMove(parameters).isolatedCopy())
{
}

AudioWorkletThread::~AudioWorkletThread() = default;

RefPtr<WorkerOrWorkletGlobalScope> AudioWorkletThread::createGlobalScope()
{
    return AudioWorkletGlobalScope::tryCreate(*this, m_parameters);
}

void AudioWorkletThread::clearProxies()
{
    m_messagingProxy = nullptr;
}

WorkerLoaderProxy* AudioWorkletThread::workerLoaderProxy()
{
    return m_messagingProxy.get();
}

WorkerDebuggerProxy* AudioWorkletThread::workerDebuggerProxy() const
{
    // FIXME: Add debugging support for AudioWorklets.
    return nullptr;
}

Ref<Thread> AudioWorkletThread::createThread()
{
    return Thread::create("WebCore: AudioWorklet"_s, [this] {
        workerOrWorkletThread();
    }, ThreadType::Audio, m_parameters.isAudioContextRealTime ? Thread::QOS::UserInteractive : Thread::QOS::Default);
}

AudioWorkletGlobalScope* AudioWorkletThread::globalScope() const
{
    return downcast<AudioWorkletGlobalScope>(WorkerOrWorkletThread::globalScope());
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
