/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

#include "Color.h"
#include "InbandGenericCue.h"
#include "TrackPrivateBase.h"
#include <wtf/JSONValues.h>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(DATACUE_VALUE)
#include "SerializedPlatformDataCue.h"
#endif

namespace WebCore {

class InbandTextTrackPrivate;
class ISOWebVTTCue;

class InbandTextTrackPrivateClient : public TrackPrivateBaseClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(InbandTextTrackPrivateClient);
public:
    virtual ~InbandTextTrackPrivateClient() = default;

    constexpr Type type() const final { return Type::Text; }

    virtual void addDataCue(const MediaTime& start, const MediaTime& end, std::span<const uint8_t>) = 0;

#if ENABLE(DATACUE_VALUE)
    virtual void addDataCue(const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&&, const String&) = 0;
    virtual void updateDataCue(const MediaTime& start, const MediaTime& end, SerializedPlatformDataCue&) = 0;
    virtual void removeDataCue(const MediaTime& start, const MediaTime& end, SerializedPlatformDataCue&) = 0;
#endif

    virtual void addGenericCue(InbandGenericCue&) = 0;
    virtual void updateGenericCue(InbandGenericCue&) = 0;
    virtual void removeGenericCue(InbandGenericCue&) = 0;

    virtual void parseWebVTTFileHeader(String&&) { ASSERT_NOT_REACHED(); }
    virtual void parseWebVTTCueData(std::span<const uint8_t>) = 0;
    virtual void parseWebVTTCueData(ISOWebVTTCue&&) = 0;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InbandTextTrackPrivateClient)
static bool isType(const WebCore::TrackPrivateBaseClient& track) { return track.type() == WebCore::TrackPrivateBaseClient::Type::Text; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
