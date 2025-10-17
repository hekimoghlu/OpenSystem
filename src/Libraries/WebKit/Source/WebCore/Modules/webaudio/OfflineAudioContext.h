/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "BaseAudioContext.h"
#include "JSDOMPromiseDeferredForward.h"
#include "OfflineAudioDestinationNode.h"
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

struct OfflineAudioContextOptions;

class OfflineAudioContext final : public BaseAudioContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(OfflineAudioContext);
public:
    static ExceptionOr<Ref<OfflineAudioContext>> create(ScriptExecutionContext&, const OfflineAudioContextOptions&);
    static ExceptionOr<Ref<OfflineAudioContext>> create(ScriptExecutionContext&, unsigned numberOfChannels, unsigned length, float sampleRate);
    void startRendering(Ref<DeferredPromise>&&);
    void suspendRendering(double suspendTime, Ref<DeferredPromise>&&);
    void resumeRendering(Ref<DeferredPromise>&&);
    void finishedRendering(bool didRendering);
    void didSuspendRendering(size_t frame);

    unsigned length() const { return m_length; }
    bool shouldSuspend();

    OfflineAudioDestinationNode& destination() final { return m_destinationNode.get(); }
    const OfflineAudioDestinationNode& destination() const final { return m_destinationNode.get(); }

private:
    OfflineAudioContext(Document&, const OfflineAudioContextOptions&);

    void lazyInitialize() final;
    void increaseNoiseMultiplierIfNeeded();

    AudioBuffer* renderTarget() const { return destination().renderTarget(); }

    // ActiveDOMObject
    bool virtualHasPendingActivity() const final;

    void settleRenderingPromise(ExceptionOr<Ref<AudioBuffer>>&&);
    void uninitialize() final;
    bool isOfflineContext() const final { return true; }

    UniqueRef<OfflineAudioDestinationNode> m_destinationNode;
    RefPtr<DeferredPromise> m_pendingRenderingPromise;
    HashMap<unsigned /* frame */, RefPtr<DeferredPromise>, IntHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> m_suspendRequests;
    unsigned m_length;
    bool m_didStartRendering { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::OfflineAudioContext)
    static bool isType(const WebCore::BaseAudioContext& context) { return context.isOfflineContext(); }
SPECIALIZE_TYPE_TRAITS_END()
