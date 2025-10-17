/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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

#include "AudioTrackPrivate.h"
#include "InbandTextTrackPrivate.h"
#include "MockBox.h"
#include "VideoTrackPrivate.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class MockAudioTrackPrivate : public AudioTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MockAudioTrackPrivate);
public:
    static Ref<MockAudioTrackPrivate> create(const MockTrackBox& box) { return adoptRef(*new MockAudioTrackPrivate(box)); }
    virtual ~MockAudioTrackPrivate() = default;

    TrackID id() const override { return m_id; }

protected:
    MockAudioTrackPrivate(const MockTrackBox& box)
        : m_box(box)
        , m_id(box.trackID())
    {
    }
    MockTrackBox m_box;
    TrackID m_id;
};

class MockTextTrackPrivate : public InbandTextTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MockTextTrackPrivate);
public:
    static Ref<MockTextTrackPrivate> create(const MockTrackBox& box) { return adoptRef(*new MockTextTrackPrivate(box)); }
    virtual ~MockTextTrackPrivate() = default;

    TrackID id() const override { return m_id; }

protected:
    MockTextTrackPrivate(const MockTrackBox& box)
        : InbandTextTrackPrivate(InbandTextTrackPrivate::CueFormat::Generic)
        , m_box(box)
        , m_id(box.trackID())
    {
    }
    MockTrackBox m_box;
    TrackID m_id;
};


class MockVideoTrackPrivate : public VideoTrackPrivate {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MockVideoTrackPrivate);
public:
    static Ref<MockVideoTrackPrivate> create(const MockTrackBox& box) { return adoptRef(*new MockVideoTrackPrivate(box)); }
    virtual ~MockVideoTrackPrivate() = default;

    TrackID id() const override { return m_id; }

protected:
    MockVideoTrackPrivate(const MockTrackBox& box)
        : m_box(box)
        , m_id(box.trackID())
    {
    }
    MockTrackBox m_box;
    TrackID m_id;
};

}

#endif
