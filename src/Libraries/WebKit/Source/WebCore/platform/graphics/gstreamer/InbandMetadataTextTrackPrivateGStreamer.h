/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "InbandTextTrackPrivate.h"

namespace WebCore {

class InbandMetadataTextTrackPrivateGStreamer : public InbandTextTrackPrivate {
public:
    static Ref<InbandMetadataTextTrackPrivateGStreamer> create(Kind kind, CueFormat cueFormat, const AtomString& id = emptyAtom())
    {
        return adoptRef(*new InbandMetadataTextTrackPrivateGStreamer(kind, cueFormat, id));
    }

    ~InbandMetadataTextTrackPrivateGStreamer() = default;

    Kind kind() const override { return m_kind; }
    std::optional<AtomString> trackUID() const override { return m_stringId; }
    AtomString inBandMetadataTrackDispatchType() const override { return m_inBandMetadataTrackDispatchType; }
    void setInBandMetadataTrackDispatchType(const AtomString& value) { m_inBandMetadataTrackDispatchType = value; }

    void addDataCue(const MediaTime& start, const MediaTime& end, std::span<const uint8_t> data)
    {
        ASSERT(isMainThread());
        ASSERT(cueFormat() == CueFormat::Data);

        notifyMainThreadClient([&](auto& client) {
            downcast<InbandTextTrackPrivateClient>(client).addDataCue(start, end, data);
        });
    }

    void addGenericCue(InbandGenericCue& data)
    {
        ASSERT(isMainThread());
        ASSERT(cueFormat() == CueFormat::Generic);
        notifyMainThreadClient([&](auto& client) {
            downcast<InbandTextTrackPrivateClient>(client).addGenericCue(data);
        });
    }

private:
    InbandMetadataTextTrackPrivateGStreamer(Kind kind, CueFormat cueFormat, const AtomString& id)
        : InbandTextTrackPrivate(cueFormat)
        , m_kind(kind)
        , m_stringId(id)
    {

    }

    Kind m_kind;
    AtomString m_stringId;
    AtomString m_inBandMetadataTrackDispatchType;
};

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
