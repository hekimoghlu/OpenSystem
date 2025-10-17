/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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

#if ENABLE(VIDEO)

#include <map>
#include <wtf/Deque.h>
#include <wtf/MediaTime.h>
#include <wtf/Observer.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/StdMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>

OBJC_CLASS AVPlayer;
OBJC_CLASS AVPlayerItem;
OBJC_CLASS AVPlayerItemVideoOutput;
OBJC_CLASS WebQueuedVideoOutputDelegate;

typedef struct __CVBuffer *CVPixelBufferRef;

namespace WebCore {

class QueuedVideoOutput
    : public RefCounted<QueuedVideoOutput>
    , public CanMakeWeakPtr<QueuedVideoOutput> {
    WTF_MAKE_TZONE_ALLOCATED(QueuedVideoOutput);
public:
    static RefPtr<QueuedVideoOutput> create(AVPlayerItem*, AVPlayer*);
    ~QueuedVideoOutput();

    bool valid();
    void invalidate();
    bool hasImageForTime(const MediaTime&) const;

    struct VideoFrameEntry {
        RetainPtr<CVPixelBufferRef> pixelBuffer;
        MediaTime displayTime;
    };
    VideoFrameEntry takeVideoFrameEntryForTime(const MediaTime&);

    void addVideoFrameEntries(Vector<VideoFrameEntry>&&);
    void purgeVideoFrameEntries();

    using CurrentImageChangedObserver = Observer<void()>;
    void addCurrentImageChangedObserver(const CurrentImageChangedObserver&);

    using ImageMap = StdMap<MediaTime, RetainPtr<CVPixelBufferRef>>;

    void rateChanged(float);

private:
    QueuedVideoOutput(AVPlayerItem*, AVPlayer*);

    void purgeImagesBeforeTime(const MediaTime&);
    void configureNextImageTimeObserver();
    void cancelNextImageTimeObserver();
    void nextImageTimeReached();

    RetainPtr<AVPlayerItem> m_playerItem;
    RetainPtr<AVPlayer> m_player;
    RetainPtr<WebQueuedVideoOutputDelegate> m_delegate;
    RetainPtr<AVPlayerItemVideoOutput> m_videoOutput;
    RetainPtr<id> m_videoTimebaseObserver;
    RetainPtr<id> m_nextImageTimebaseObserver;

    ImageMap m_videoFrames;
    WeakHashSet<CurrentImageChangedObserver> m_currentImageChangedObservers;

    bool m_paused { true };
};

}

#endif
