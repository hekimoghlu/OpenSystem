/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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

#include <CoreMedia/CMTime.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS AVSampleBufferAudioRenderer;
OBJC_CLASS AVSampleBufferDisplayLayer;
OBJC_CLASS NSError;
OBJC_CLASS WebAVSampleBufferListenerPrivate;
OBJC_PROTOCOL(WebSampleBufferVideoRendering);

namespace WebCore {
class WebAVSampleBufferListenerClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::WebAVSampleBufferListenerClient> : std::true_type { };
}

namespace WebCore {

class WebAVSampleBufferListenerClient : public CanMakeWeakPtr<WebAVSampleBufferListenerClient> {
public:
    virtual ~WebAVSampleBufferListenerClient() = default;
    virtual void videoRendererDidReceiveError(WebSampleBufferVideoRendering *, NSError *) { }
    virtual void videoRendererRequiresFlushToResumeDecodingChanged(WebSampleBufferVideoRendering *, bool) { }
    virtual void videoRendererReadyForDisplayChanged(WebSampleBufferVideoRendering *, bool) { }
    virtual void audioRendererDidReceiveError(AVSampleBufferAudioRenderer *, NSError *) { }
    virtual void audioRendererWasAutomaticallyFlushed(AVSampleBufferAudioRenderer *, const CMTime&) { }
    virtual void outputObscuredDueToInsufficientExternalProtectionChanged(bool) { }
};

class WebAVSampleBufferListener final : public ThreadSafeRefCounted<WebAVSampleBufferListener> {
public:
    static Ref<WebAVSampleBufferListener> create(WebAVSampleBufferListenerClient& client) { return adoptRef(*new WebAVSampleBufferListener(client)); }
    void invalidate();
    void beginObservingVideoRenderer(WebSampleBufferVideoRendering *);
    void stopObservingVideoRenderer(WebSampleBufferVideoRendering *);
    void beginObservingAudioRenderer(AVSampleBufferAudioRenderer *);
    void stopObservingAudioRenderer(AVSampleBufferAudioRenderer *);
private:
    explicit WebAVSampleBufferListener(WebAVSampleBufferListenerClient&);
    RetainPtr<WebAVSampleBufferListenerPrivate> m_private;
};

} // namespace WebCore
