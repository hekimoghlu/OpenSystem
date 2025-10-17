/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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

#if ENABLE(WIRELESS_PLAYBACK_TARGET)

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class MediaPlaybackTargetContextMockState : uint8_t {
    Unknown = 0,
    OutputDeviceUnavailable = 1,
    OutputDeviceAvailable = 2,
};

enum class MediaPlaybackTargetContextType : uint8_t {
    AVOutputContext,
    Mock,
    Serialized,
};

class MediaPlaybackTargetContext {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MediaPlaybackTargetContext);
public:
    using Type = MediaPlaybackTargetContextType;
    using MockState = MediaPlaybackTargetContextMockState;

    WEBCORE_EXPORT virtual ~MediaPlaybackTargetContext() = default;

    Type type() const { return m_type; }
    WEBCORE_EXPORT virtual String deviceName() const = 0;
    WEBCORE_EXPORT virtual bool hasActiveRoute() const = 0;
    WEBCORE_EXPORT virtual bool supportsRemoteVideoPlayback() const = 0;

protected:
    MediaPlaybackTargetContext(Type type)
        : m_type(type)
    {
    }

private:
    // This should be const, however IPC's Decoder's handling doesn't allow for const member.
    Type m_type;
};

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
