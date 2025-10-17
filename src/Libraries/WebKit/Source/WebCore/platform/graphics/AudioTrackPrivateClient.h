/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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

#include "TrackPrivateBaseClient.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(VIDEO)

namespace WebCore {

class AudioTrackPrivate;
struct PlatformAudioTrackConfiguration;

class AudioTrackPrivateClient : public TrackPrivateBaseClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(AudioTrackPrivateClient);
public:
    constexpr Type type() const final { return Type::Audio; }
    virtual void enabledChanged(bool) = 0;
    virtual void configurationChanged(const PlatformAudioTrackConfiguration&) = 0;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioTrackPrivateClient)
static bool isType(const WebCore::TrackPrivateBaseClient& track) { return track.type() == WebCore::TrackPrivateBaseClient::Type::Audio; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
