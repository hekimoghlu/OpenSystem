/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

#if ENABLE(MEDIA_USAGE)

#include "MediaUsageManager.h"
#include <wtf/TZoneMallocInlines.h>

OBJC_CLASS USVideoUsage;

namespace WebKit {

class MediaUsageManagerCocoa : public MediaUsageManager {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MediaUsageManagerCocoa);
public:
    MediaUsageManagerCocoa() = default;
    virtual ~MediaUsageManagerCocoa();

private:
    void reset() final;
    void addMediaSession(WebCore::MediaSessionIdentifier, const String&, const URL&) final;
    void updateMediaUsage(WebCore::MediaSessionIdentifier, const WebCore::MediaUsageInfo&) final;
    void removeMediaSession(WebCore::MediaSessionIdentifier) final;
#if !HAVE(CGS_FIX_FOR_RADAR_97530095)
    bool isPlayingVideoInViewport() const final;
#endif

    struct SessionMediaUsage {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        SessionMediaUsage(WebCore::MediaSessionIdentifier identifier, const String& bundleIdentifier, const URL& pageURL)
            : identifier(identifier)
            , bundleIdentifier(bundleIdentifier)
            , pageURL(pageURL)
        {
        }

        WebCore::MediaSessionIdentifier identifier;
        String bundleIdentifier;
        URL pageURL;
        std::optional<WebCore::MediaUsageInfo> mediaUsageInfo;
        RetainPtr<USVideoUsage> usageTracker;
    };

    HashMap<WebCore::MediaSessionIdentifier, std::unique_ptr<SessionMediaUsage>> m_mediaSessions;
};

} // namespace WebKit

#endif // ENABLE(MEDIA_USAGE)

