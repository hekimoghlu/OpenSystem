/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#include <wtf/RetainPtr.h>

OBJC_CLASS AVOutputContext;

namespace WebCore {

class WEBCORE_EXPORT MediaPlaybackTargetContextCocoa final : public MediaPlaybackTargetContext {
public:
    explicit MediaPlaybackTargetContextCocoa(RetainPtr<AVOutputContext>&&);
    ~MediaPlaybackTargetContextCocoa();

    RetainPtr<AVOutputContext> outputContext() const;
private:
    String deviceName() const final;
    bool hasActiveRoute() const final;
    bool supportsRemoteVideoPlayback() const final;

    RetainPtr<AVOutputContext> m_outputContext;
};

class MediaPlaybackTargetCocoa final : public MediaPlaybackTarget {
public:
    WEBCORE_EXPORT static Ref<MediaPlaybackTarget> create(MediaPlaybackTargetContextCocoa&&);

#if PLATFORM(IOS_FAMILY) && !PLATFORM(IOS_FAMILY_SIMULATOR) && !PLATFORM(MACCATALYST)
    WEBCORE_EXPORT static Ref<MediaPlaybackTargetCocoa> create();
#endif

    TargetType targetType() const final { return TargetType::AVFoundation; }
    const MediaPlaybackTargetContext& targetContext() const final { return m_context; }

    AVOutputContext* outputContext() { return m_context.outputContext().get(); }

private:
    explicit MediaPlaybackTargetCocoa(MediaPlaybackTargetContextCocoa&&);

    MediaPlaybackTargetContextCocoa m_context;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaPlaybackTargetContextCocoa)
static bool isType(const WebCore::MediaPlaybackTargetContext& context)
{
    return context.type() ==  WebCore::MediaPlaybackTargetContextType::AVOutputContext;
}
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaPlaybackTargetCocoa)
static bool isType(const WebCore::MediaPlaybackTarget& target)
{
    return target.targetType() ==  WebCore::MediaPlaybackTarget::TargetType::AVFoundation;
}
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
