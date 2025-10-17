/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#include "WorkerLoaderProxy.h"
#include "WorkletGlobalScopeProxy.h"

namespace WebCore {

class AudioWorklet;
class AudioWorkletThread;
class Document;

class AudioWorkletMessagingProxy final : public WorkletGlobalScopeProxy, public WorkerLoaderProxy, public CanMakeThreadSafeCheckedPtr<AudioWorkletMessagingProxy> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(AudioWorkletMessagingProxy);
public:
    static Ref<AudioWorkletMessagingProxy> create(AudioWorklet& worklet)
    {
        return adoptRef(*new AudioWorkletMessagingProxy(worklet));
    }

    ~AudioWorkletMessagingProxy();

    // This method is used in the main thread to post task back to the worklet thread.
    bool postTaskForModeToWorkletGlobalScope(ScriptExecutionContext::Task&&, const String& mode) final;

    AudioWorkletThread& workletThread() { return m_workletThread.get(); }

    void postTaskToAudioWorklet(Function<void(AudioWorklet&)>&&);
    ScriptExecutionContextIdentifier loaderContextIdentifier() const final;

    uint32_t checkedPtrCount() const final { return CanMakeThreadSafeCheckedPtr<AudioWorkletMessagingProxy>::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeThreadSafeCheckedPtr<AudioWorkletMessagingProxy>::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeThreadSafeCheckedPtr<AudioWorkletMessagingProxy>::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeThreadSafeCheckedPtr<AudioWorkletMessagingProxy>::decrementCheckedPtrCount(); }

private:
    explicit AudioWorkletMessagingProxy(AudioWorklet&);

    // WorkerLoaderProxy.
    RefPtr<CacheStorageConnection> createCacheStorageConnection() final;
    RefPtr<RTCDataChannelRemoteHandlerConnection> createRTCDataChannelRemoteHandlerConnection() final;
    void postTaskToLoader(ScriptExecutionContext::Task&&) final;

    bool isAudioWorkletMessagingProxy() const final { return true; }

    WeakPtr<AudioWorklet> m_worklet;
    Ref<Document> m_document;
    Ref<AudioWorkletThread> m_workletThread;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioWorkletMessagingProxy)
static bool isType(const WebCore::WorkletGlobalScopeProxy& proxy) { return proxy.isAudioWorkletMessagingProxy(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_AUDIO)
