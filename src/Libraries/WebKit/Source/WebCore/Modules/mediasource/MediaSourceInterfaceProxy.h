/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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

#include "PlatformTimeRanges.h"
#include <wtf/MediaTime.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class HTMLMediaElement;
class MediaSourcePrivateClient;
class TimeRanges;

class MediaSourceInterfaceProxy
    : public RefCounted<MediaSourceInterfaceProxy>
    , public CanMakeWeakPtr<MediaSourceInterfaceProxy> {

public:
    virtual ~MediaSourceInterfaceProxy() = default;

    virtual RefPtr<MediaSourcePrivateClient> client() const = 0;
    virtual void monitorSourceBuffers() = 0;
    virtual bool isClosed() const = 0;
    virtual MediaTime duration() const = 0;
    virtual PlatformTimeRanges buffered() const = 0;
    virtual Ref<TimeRanges> seekable() const = 0;
    virtual bool isStreamingContent() const = 0;
    virtual bool attachToElement(WeakPtr<HTMLMediaElement>&&) = 0;
    virtual void detachFromElement() = 0;
    virtual void elementIsShuttingDown() = 0;
    virtual void openIfDeferredOpen() = 0;
    virtual bool isManaged() const = 0;
    virtual void setAsSrcObject(bool) = 0;
    virtual void memoryPressure() = 0;
    virtual bool detachable() const = 0;
};

} // namespace WebCore

#endif
