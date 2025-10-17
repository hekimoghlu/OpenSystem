/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
#include "config.h"
#include "MediaPlaybackTargetContextSerialized.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET)

#include "WKKeyedCoder.h"

#include <pal/spi/cocoa/AVFoundationSPI.h>
#include <pal/cocoa/AVFoundationSoftLink.h>

using namespace WebCore;

namespace WebKit {

MediaPlaybackTargetContextSerialized::MediaPlaybackTargetContextSerialized(const MediaPlaybackTargetContext& context)
    : MediaPlaybackTargetContext(MediaPlaybackTargetContextType::Serialized)
    , m_deviceName(context.deviceName())
    , m_hasActiveRoute(context.hasActiveRoute())
    , m_supportsRemoteVideoPlayback(context.supportsRemoteVideoPlayback())
    , m_targetType(is<MediaPlaybackTargetContextSerialized>(context) ? downcast<MediaPlaybackTargetContextSerialized>(context).targetType() : context.type())
{
    if (is<MediaPlaybackTargetContextCocoa>(context)) {
        auto archiver = adoptNS([WKKeyedCoder new]);
        [downcast<MediaPlaybackTargetContextCocoa>(context).outputContext() encodeWithCoder:archiver.get()];
        auto dictionary = [archiver accumulatedDictionary];
        m_contextID = (NSString *)[dictionary objectForKey:@"AVOutputContextSerializationKeyContextID"];
        m_contextType = (NSString *)[dictionary objectForKey:@"AVOutputContextSerializationKeyContextType"];
    } else if (is<MediaPlaybackTargetContextMock>(context))
        m_state = downcast<MediaPlaybackTargetContextMock>(context).state();
    else if (is<MediaPlaybackTargetContextSerialized>(context)) {
        m_state = downcast<MediaPlaybackTargetContextSerialized>(context).m_state;
        m_contextID = downcast<MediaPlaybackTargetContextSerialized>(context).m_contextID;
        m_contextType = downcast<MediaPlaybackTargetContextSerialized>(context).m_contextType;
    }
}

MediaPlaybackTargetContextSerialized::MediaPlaybackTargetContextSerialized(String&& deviceName, bool hasActiveRoute, bool supportsRemoteVideoPlayback, MediaPlaybackTargetContextType targetType, MediaPlaybackTargetContextMockState state, String&& contextID, String&& contextType)
    : MediaPlaybackTargetContext(MediaPlaybackTargetContextType::Serialized)
    , m_deviceName(WTFMove(deviceName))
    , m_hasActiveRoute(hasActiveRoute)
    , m_supportsRemoteVideoPlayback(supportsRemoteVideoPlayback)
    , m_targetType(targetType)
    , m_state(state)
    , m_contextID(WTFMove(contextID))
    , m_contextType(WTFMove(contextType))
{
}

std::variant<MediaPlaybackTargetContextCocoa, MediaPlaybackTargetContextMock> MediaPlaybackTargetContextSerialized::platformContext() const
{
    if (m_targetType == MediaPlaybackTargetContextType::Mock)
        return MediaPlaybackTargetContextMock(m_deviceName, m_state);

    ASSERT(m_targetType == MediaPlaybackTargetContextType::AVOutputContext);

    auto propertyList = [NSMutableDictionary dictionaryWithCapacity:2];
    propertyList[@"AVOutputContextSerializationKeyContextID"] = m_contextID;
    propertyList[@"AVOutputContextSerializationKeyContextType"] = m_contextType;
    auto unarchiver = adoptNS([[WKKeyedCoder alloc] initWithDictionary:propertyList]);
    auto outputContext = adoptNS([[PAL::getAVOutputContextClass() alloc] initWithCoder:unarchiver.get()]);
    // std::variant construction in older clang gives either an error, a vtable linkage error unless we construct it this way.
    std::variant<MediaPlaybackTargetContextCocoa, MediaPlaybackTargetContextMock> variant { std::in_place_type<MediaPlaybackTargetContextCocoa>, WTFMove(outputContext) };
    return variant;
}

} // namespace WebKit

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
