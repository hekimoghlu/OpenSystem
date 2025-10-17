/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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

#include <WebCore/MediaPlaybackTargetCocoa.h>
#include <WebCore/MediaPlaybackTargetMock.h>
#include <variant>

namespace WebKit {

class MediaPlaybackTargetContextSerialized final : public WebCore::MediaPlaybackTargetContext {
public:
    explicit MediaPlaybackTargetContextSerialized(const WebCore::MediaPlaybackTargetContext&);

    String deviceName() const final { return m_deviceName; }
    bool hasActiveRoute() const final { return m_hasActiveRoute; }
    bool supportsRemoteVideoPlayback() const { return m_supportsRemoteVideoPlayback; }

    std::variant<WebCore::MediaPlaybackTargetContextCocoa, WebCore::MediaPlaybackTargetContextMock> platformContext() const;

    // Used by IPC serializer.
    WebCore::MediaPlaybackTargetContextType targetType() const { return m_targetType; }
    WebCore::MediaPlaybackTargetContextMockState mockState() const { return m_state; }
    String contextID() const { return m_contextID; }
    String contextType() const { return m_contextType; }
    MediaPlaybackTargetContextSerialized(String&&, bool, bool, WebCore::MediaPlaybackTargetContextType, WebCore::MediaPlaybackTargetContextMockState, String&&, String&&);

private:
    String m_deviceName;
    bool m_hasActiveRoute { false };
    bool m_supportsRemoteVideoPlayback { false };
    // This should be const, however IPC's Decoder's handling doesn't allow for const member.
    WebCore::MediaPlaybackTargetContextType m_targetType;
    WebCore::MediaPlaybackTargetContextMockState m_state { WebCore::MediaPlaybackTargetContextMockState::Unknown };
    String m_contextID;
    String m_contextType;
};

class MediaPlaybackTargetSerialized final : public WebCore::MediaPlaybackTarget {
public:
    static Ref<MediaPlaybackTarget> create(MediaPlaybackTargetContextSerialized&& context)
    {
        return adoptRef(*new MediaPlaybackTargetSerialized(WTFMove(context)));
    }

    TargetType targetType() const final { return TargetType::Serialized; }
    const WebCore::MediaPlaybackTargetContext& targetContext() const final { return m_context; }

private:
    explicit MediaPlaybackTargetSerialized(MediaPlaybackTargetContextSerialized&& context)
        : m_context(WTFMove(context))
    {
    }

    MediaPlaybackTargetContextSerialized m_context;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::MediaPlaybackTargetContextSerialized)
static bool isType(const WebCore::MediaPlaybackTargetContext& context)
{
    return context.type() ==  WebCore::MediaPlaybackTargetContextType::Serialized;
}
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::MediaPlaybackTargetSerialized)
static bool isType(const WebCore::MediaPlaybackTarget& target)
{
    return target.targetType() ==  WebCore::MediaPlaybackTarget::TargetType::Serialized;
}
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
