/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

#if ENABLE(MEDIA_SOURCE)

#include "MediaSourceInterfaceProxy.h"

namespace WebCore {

class MediaSource;

class MediaSourceInterfaceMainThread : public MediaSourceInterfaceProxy {
public:
    static Ref<MediaSourceInterfaceMainThread> create(Ref<MediaSource>&& mediaSource) { return adoptRef(*new MediaSourceInterfaceMainThread(WTFMove(mediaSource))); }

private:
    RefPtr<MediaSourcePrivateClient> client() const final;
    void monitorSourceBuffers() final;
    bool isClosed() const final;
    MediaTime duration() const final;
    PlatformTimeRanges buffered() const final;
    Ref<TimeRanges> seekable() const final;
    bool isStreamingContent() const final;
    bool attachToElement(WeakPtr<HTMLMediaElement>&&) final;
    void detachFromElement() final;
    void elementIsShuttingDown() final;
    void openIfDeferredOpen() final;
    bool isManaged() const final;
    void setAsSrcObject(bool) final;
    void memoryPressure() final;
    bool detachable() const final;

    explicit MediaSourceInterfaceMainThread(Ref<MediaSource>&&);

    Ref<MediaSource> m_mediaSource;
};

} // namespace WebCore

#endif
