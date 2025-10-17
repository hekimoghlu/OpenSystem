/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "MediaPlaybackTarget.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class MediaPlaybackTargetContextMock final : public MediaPlaybackTargetContext {
public:
    using State = MediaPlaybackTargetContextMockState;

    MediaPlaybackTargetContextMock(const String& mockDeviceName, State mockState)
        : MediaPlaybackTargetContext(Type::Mock)
        , m_mockDeviceName(mockDeviceName)
        , m_mockState(mockState)
    {
    }

    State state() const
    {
        return m_mockState;
    }

    String deviceName() const final { return m_mockDeviceName; }
    bool hasActiveRoute() const final { return !m_mockDeviceName.isEmpty(); }
    bool supportsRemoteVideoPlayback() const final { return !m_mockDeviceName.isEmpty(); }

private:
    String m_mockDeviceName;
    State m_mockState { State::Unknown };
};

class MediaPlaybackTargetMock final : public MediaPlaybackTarget {
public:
    WEBCORE_EXPORT static Ref<MediaPlaybackTarget> create(MediaPlaybackTargetContextMock&&);

    MediaPlaybackTargetContextMock::State state() const { return m_context.state(); }

private:
    explicit MediaPlaybackTargetMock(MediaPlaybackTargetContextMock&&);
    TargetType targetType() const final { return MediaPlaybackTarget::TargetType::Mock; }
    const MediaPlaybackTargetContext& targetContext() const final { return m_context; }

    MediaPlaybackTargetContextMock m_context;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaPlaybackTargetContextMock)
static bool isType(const WebCore::MediaPlaybackTargetContext& context)
{
    return context.type() ==  WebCore::MediaPlaybackTargetContextType::Mock;
}
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaPlaybackTargetMock)
static bool isType(const WebCore::MediaPlaybackTarget& target)
{
    return target.targetType() ==  WebCore::MediaPlaybackTarget::TargetType::Mock;
}
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
