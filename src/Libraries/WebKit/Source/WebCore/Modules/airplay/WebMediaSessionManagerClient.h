/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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
#include "MediaProducer.h"
#include "PlatformView.h"
#include "PlaybackTargetClientContextIdentifier.h"
#include <wtf/Ref.h>
#include <wtf/RetainPtr.h>

namespace WebCore {

class WebMediaSessionManagerClient {
public:
    virtual ~WebMediaSessionManagerClient() = default;

    virtual void setPlaybackTarget(PlaybackTargetClientContextIdentifier, Ref<MediaPlaybackTarget>&&) = 0;
    virtual void externalOutputDeviceAvailableDidChange(PlaybackTargetClientContextIdentifier, bool) = 0;
    virtual void setShouldPlayToPlaybackTarget(PlaybackTargetClientContextIdentifier, bool) = 0;
    virtual void playbackTargetPickerWasDismissed(PlaybackTargetClientContextIdentifier) = 0;
    virtual bool alwaysOnLoggingAllowed() const { return false; }
    virtual RetainPtr<PlatformView> platformView() const = 0;
};

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
