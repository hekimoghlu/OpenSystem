/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#include "WebMediaSessionManagerMac.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#include "MediaPlaybackTargetPickerMac.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

WebMediaSessionManager& WebMediaSessionManager::shared()
{
    static NeverDestroyed<WebMediaSessionManagerMac> sharedManager;
    return sharedManager;
}

WebMediaSessionManagerMac::WebMediaSessionManagerMac()
    : WebMediaSessionManager()
{
}

WebMediaSessionManagerMac::~WebMediaSessionManagerMac() = default;

WebCore::MediaPlaybackTargetPicker& WebMediaSessionManagerMac::platformPicker()
{
    if (!m_targetPicker)
        m_targetPicker = makeUnique<MediaPlaybackTargetPickerMac>(*this);

    return *m_targetPicker.get();
}

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)
