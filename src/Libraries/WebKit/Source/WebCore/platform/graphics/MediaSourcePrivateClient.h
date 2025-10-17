/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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

#include "MediaPromiseTypes.h"
#include "PlatformTimeRanges.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/Logger.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class MediaSourcePrivate;

class MediaSourcePrivateClient : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<MediaSourcePrivateClient> {
public:
    virtual ~MediaSourcePrivateClient() = default;

    virtual void setPrivateAndOpen(Ref<MediaSourcePrivate>&&) = 0;
    virtual void reOpen() = 0;
    virtual Ref<MediaTimePromise> waitForTarget(const SeekTarget&) = 0;
    virtual Ref<MediaPromise> seekToTime(const MediaTime&) = 0;
    virtual RefPtr<MediaSourcePrivate> mediaSourcePrivate() const = 0;

#if !RELEASE_LOG_DISABLED
    virtual void setLogIdentifier(uint64_t) = 0;
    virtual const Logger* logger() const { return nullptr; }
#endif

    enum class RendererType { Audio, Video };
    virtual void failedToCreateRenderer(RendererType) = 0;
};

}

#endif // ENABLE(MEDIA_SOURCE)
