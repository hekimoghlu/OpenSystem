/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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

#include "PlatformVideoTrackConfiguration.h"
#include "TrackPrivateBase.h"
#include "VideoTrackPrivateClient.h"
#include <wtf/Function.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

struct VideoInfo;

class VideoTrackPrivate : public TrackPrivateBase {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(VideoTrackPrivate);
public:
    virtual void setSelected(bool selected)
    {
        if (m_selected == selected)
            return;
        m_selected = selected;
        notifyClients([selected](auto& client) {
            downcast<VideoTrackPrivateClient>(client).selectedChanged(selected);
        });
        if (m_selectedChangedCallback)
            m_selectedChangedCallback(*this, m_selected);
    }
    virtual bool selected() const { return m_selected; }

    enum class Kind : uint8_t { Alternative, Captions, Main, Sign, Subtitles, Commentary, None };
    virtual Kind kind() const { return Kind::None; }

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "VideoTrackPrivate"_s; }
#endif

    using SelectedChangedCallback = Function<void(VideoTrackPrivate&, bool selected)>;
    void setSelectedChangedCallback(SelectedChangedCallback&& callback) { m_selectedChangedCallback = WTFMove(callback); }

    const PlatformVideoTrackConfiguration& configuration() const { return m_configuration; }
    void setConfiguration(PlatformVideoTrackConfiguration&& configuration)
    {
        if (configuration == m_configuration)
            return;
        m_configuration = WTFMove(configuration);
        notifyClients([configuration = m_configuration](auto& client) {
            downcast<VideoTrackPrivateClient>(client).configurationChanged(configuration);
        });
    }
    
    bool operator==(const VideoTrackPrivate& track) const
    {
        return TrackPrivateBase::operator==(track)
            && configuration() == track.configuration()
            && kind() == track.kind();
    }

    Type type() const final { return Type::Video; }

    virtual void setFormatDescription(Ref<VideoInfo>&&) { }

protected:
    VideoTrackPrivate() = default;

private:
    bool m_selected { false };
    PlatformVideoTrackConfiguration m_configuration;

    SelectedChangedCallback m_selectedChangedCallback;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::VideoTrackPrivate)
static bool isType(const WebCore::TrackPrivateBase& track) { return track.type() == WebCore::TrackPrivateBase::Type::Video; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
