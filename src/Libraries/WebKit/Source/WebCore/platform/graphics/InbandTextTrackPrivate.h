/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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

#include "InbandTextTrackPrivateClient.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

enum class InbandTextTrackPrivateMode : uint8_t {
    Disabled,
    Hidden,
    Showing
};

class InbandTextTrackPrivate : public TrackPrivateBase {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(InbandTextTrackPrivate);
public:
    enum class CueFormat : uint8_t {
        Data,
        Generic,
        WebVTT
    };
    static Ref<InbandTextTrackPrivate> create(CueFormat format) { return adoptRef(*new InbandTextTrackPrivate(format)); }
    virtual ~InbandTextTrackPrivate() = default;

    using Mode = InbandTextTrackPrivateMode;
    virtual void setMode(Mode mode) { m_mode = mode; };
    virtual InbandTextTrackPrivate::Mode mode() const { return m_mode; }

    enum class Kind : uint8_t {
        Subtitles,
        Captions,
        Descriptions,
        Chapters,
        Metadata,
        Forced,
        None
    };
    virtual Kind kind() const { return Kind::Subtitles; }

    virtual bool isClosedCaptions() const { return false; }
    virtual bool isSDH() const { return false; }
    virtual bool containsOnlyForcedSubtitles() const { return false; }
    virtual bool isMainProgramContent() const { return true; }
    virtual bool isEasyToRead() const { return false; }
    virtual bool isDefault() const { return false; }
    AtomString label() const override { return emptyAtom(); }
    AtomString language() const override { return emptyAtom(); }
    std::optional<AtomString> trackUID() const override { return emptyAtom(); }
    virtual AtomString inBandMetadataTrackDispatchType() const { return emptyAtom(); }

    CueFormat cueFormat() const { return m_format; }
    
    bool operator==(const InbandTextTrackPrivate& track) const
    {
        return TrackPrivateBase::operator==(track)
            && cueFormat() == track.cueFormat()
            && kind() == track.kind()
            && isClosedCaptions() == track.isClosedCaptions()
            && isSDH() == track.isSDH()
            && containsOnlyForcedSubtitles() == track.containsOnlyForcedSubtitles()
            && isMainProgramContent() == track.isMainProgramContent()
            && isEasyToRead() == track.isEasyToRead()
            && isDefault() == track.isDefault()
            && inBandMetadataTrackDispatchType() == track.inBandMetadataTrackDispatchType();
    }

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const override { return "InbandTextTrackPrivate"_s; }
#endif

    Type type() const final { return Type::Text; };

protected:
    InbandTextTrackPrivate(CueFormat format)
        : m_format(format)
    {
    }

private:
    CueFormat m_format;
    Mode m_mode { Mode::Disabled };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InbandTextTrackPrivate)
static bool isType(const WebCore::TrackPrivateBase& track) { return track.type() == WebCore::TrackPrivateBase::Type::Text; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)

