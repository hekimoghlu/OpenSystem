/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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

#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class TextTrack;
class InbandTextTrackPrivate;

struct PlatformTextTrackData {
    enum class TrackKind : uint8_t {
        Subtitle = 0,
        Caption = 1,
        Description = 2,
        Chapter = 3,
        MetaData = 4,
        Forced = 5,
    };
    enum class TrackType : uint8_t {
        InBand = 0,
        OutOfBand = 1,
        Script = 2
    };
    enum class TrackMode : uint8_t {
        Disabled,
        Hidden,
        Showing
    };

    PlatformTextTrackData() = default;
    PlatformTextTrackData(const String& label, const String& language, const String& url, TrackMode mode, TrackKind kind, TrackType type, int uniqueId, bool isDefault)
        : m_label(label)
        , m_language(language)
        , m_url(url)
        , m_mode(mode)
        , m_kind(kind)
        , m_type(type)
        , m_uniqueId(uniqueId)
        , m_isDefault(isDefault)
    {
    }

    String m_label;
    String m_language;
    String m_url;
    TrackMode m_mode;
    TrackKind m_kind;
    TrackType m_type;
    int m_uniqueId;
    bool m_isDefault;
};

class PlatformTextTrackClient {
public:
    virtual ~PlatformTextTrackClient() = default;
    
    virtual TextTrack* publicTrack() = 0;
    virtual InbandTextTrackPrivate* privateTrack() { return 0; }
};

class PlatformTextTrack : public RefCounted<PlatformTextTrack> {
public:
    static Ref<PlatformTextTrack> create(PlatformTextTrackClient* client, const String& label, const String& language, PlatformTextTrackData::TrackMode mode, PlatformTextTrackData::TrackKind kind, PlatformTextTrackData::TrackType type, int uniqueId)
    {
        return adoptRef(*new PlatformTextTrack(client, label, language, String(), mode, kind, type, uniqueId, false));
    }

    static Ref<PlatformTextTrack> createOutOfBand(const String& label, const String& language, const String& url, PlatformTextTrackData::TrackMode mode, PlatformTextTrackData::TrackKind kind, int uniqueId, bool isDefault)
    {
        return adoptRef(*new PlatformTextTrack(nullptr, label, language, url, mode, kind, PlatformTextTrackData::TrackType::OutOfBand, uniqueId, isDefault));
    }
    
    static Ref<PlatformTextTrack> create(PlatformTextTrackData&& data)
    {
        return adoptRef(*new PlatformTextTrack(WTFMove(data)));
    }

    virtual ~PlatformTextTrack() = default;
    
    PlatformTextTrackData::TrackType type() const { return m_trackData.m_type; }
    PlatformTextTrackData::TrackKind kind() const { return m_trackData.m_kind; }
    PlatformTextTrackData::TrackMode mode() const { return m_trackData.m_mode; }
    const String& label() const { return m_trackData.m_label; }
    const String& language() const { return m_trackData.m_language; }
    const String& url() const { return m_trackData.m_url; }
    int uniqueId() const { return m_trackData.m_uniqueId; }
    bool isDefault() const { return m_trackData.m_isDefault; }
    PlatformTextTrackClient* client() const { return m_client; }
    
    PlatformTextTrackData data() const { return m_trackData; }

protected:
    PlatformTextTrack(PlatformTextTrackClient* client, const String& label, const String& language, const String& url, PlatformTextTrackData::TrackMode mode, PlatformTextTrackData::TrackKind kind, PlatformTextTrackData::TrackType type, int uniqueId, bool isDefault)
        : m_client(client)
    {
        m_trackData = {
            label,
            language,
            url,
            mode,
            kind,
            type,
            uniqueId,
            isDefault,
        };
    }
    
    PlatformTextTrack(PlatformTextTrackData&& data)
        : m_trackData(WTFMove(data))
    {
    }

    PlatformTextTrackData m_trackData;
    PlatformTextTrackClient* m_client;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
