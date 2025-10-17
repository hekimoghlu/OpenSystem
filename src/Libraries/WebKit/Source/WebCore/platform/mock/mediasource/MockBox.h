/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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
#ifndef MockBox_h
#define MockBox_h

#if ENABLE(MEDIA_SOURCE)

#include <wtf/MediaTime.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {

// A MockBox represents a ISO-BMFF like data structure. Data in the
// structure is little-endian. The layout of the data structure as follows:
//
// 4 bytes : 4CC : identifier
// 4 bytes : unsigned : length
// 
class MockBox {
public:
    static String peekType(JSC::ArrayBuffer*);
    static size_t peekLength(JSC::ArrayBuffer*);

    size_t length() const { return m_length; }
    const String& type() const { return m_type; }

protected:
    MockBox(JSC::ArrayBuffer*);

    size_t m_length;
    String m_type;
};

// A MockTrackBox extends MockBox and expects the following
// data structure:
//
// 4 bytes : 4CC : identifier = 'trak'
// 4 bytes : unsigned : length = 17
// 4 bytes : signed : track ID
// 4 bytes : 4CC : codec
// 1 byte  : unsigned : kind
// 
class MockTrackBox final : public MockBox {
public:
    static const String& type();
    MockTrackBox(JSC::ArrayBuffer*);

    int32_t trackID() const { return m_trackID; }

    const String& codec() const { return m_codec; }

    enum TrackKind { Audio, Video, Text };
    TrackKind kind() const { return m_kind; }

private:
    uint8_t m_trackID;
    String m_codec;
    TrackKind m_kind;
};

// A MockInitializationBox extends MockBox and contains 0 or more
// MockTrackBoxes. It expects the following data structure:
//
// 4 bytes : 4CC : identifier = 'init'
// 4 bytes : unsigned : length = 16 + (13 * num tracks)
// 4 bytes : signed : duration time value
// 4 bytes : signed : duration time scale
// N bytes : MockTrackBoxes : tracks
//
class MockInitializationBox final : public MockBox {
public:
    static const String& type();
    MockInitializationBox(JSC::ArrayBuffer*);

    MediaTime duration() const { return m_duration; }
    const Vector<MockTrackBox>& tracks() const { return m_tracks; }

private:
    MediaTime m_duration;
    Vector<MockTrackBox> m_tracks;
};

// A MockSampleBox extends MockBox and expects the following data structure:
//
// 4 bytes : 4CC : identifier = 'smpl'
// 4 bytes : unsigned : length = 29
// 4 bytes : signed : time scale
// 4 bytes : signed : presentation time value
// 4 bytes : signed : decode time value
// 4 bytes : signed : duration time value
// 4 bytes : signed : track ID
// 1 byte  : unsigned : flags
// 1 byte  : unsigned : generation
//
class MockSampleBox final : public MockBox {
public:
    static const String& type();
    MockSampleBox(JSC::ArrayBuffer*);

    MediaTime presentationTimestamp() const { return m_presentationTimestamp; }
    MediaTime decodeTimestamp() const { return m_decodeTimestamp; }
    MediaTime duration() const { return m_duration; }
    int32_t trackID() const { return m_trackID; }
    uint8_t flags() const { return m_flags; }
    uint8_t generation() const { return m_generation; }
    void offsetTimestampsBy(const MediaTime& offset)
    {
        m_presentationTimestamp += offset;
        m_decodeTimestamp += offset;
    }
    void setTimestamps(const MediaTime& presentationTimestamp, const MediaTime& decodeTimestamp)
    {
        m_presentationTimestamp = presentationTimestamp;
        m_decodeTimestamp = decodeTimestamp;
    }

    void clearFlag(uint8_t flag) { m_flags &= ~flag; }
    void setFlag(uint8_t flag) { m_flags |= flag; }

    enum {
        IsSync = 1 << 0,
        IsCorrupted = 1 << 1,
        IsDropped = 1 << 2,
        IsDelayed = 1 << 3,
        IsNonDisplaying = 1 << 4,
    };
    bool isSync() const { return m_flags & IsSync; }
    bool isCorrupted() const { return m_flags & IsCorrupted; }
    bool isDropped() const { return m_flags & IsDropped; }
    bool isDelayed() const { return m_flags & IsDelayed; }
    bool isNonDisplaying() const { return m_flags & IsNonDisplaying; }

private:
    MediaTime m_presentationTimestamp;
    MediaTime m_decodeTimestamp;
    MediaTime m_duration;
    int32_t m_trackID;
    uint8_t m_flags;
    uint8_t m_generation;
};

}

#endif

#endif
