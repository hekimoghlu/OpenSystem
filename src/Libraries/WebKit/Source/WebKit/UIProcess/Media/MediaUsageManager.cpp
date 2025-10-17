/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#include "MediaUsageManager.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MEDIA_USAGE)

#include <WebCore/NotImplemented.h>

namespace WebKit {

#if !HAVE(MEDIA_USAGE_FRAMEWORK)
std::unique_ptr<MediaUsageManager> MediaUsageManager::create()
{
    return makeUnique<MediaUsageManager>();
}
#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaUsageManager);

void MediaUsageManager::reset()
{
    notImplemented();
}

void MediaUsageManager::addMediaSession(WebCore::MediaSessionIdentifier identifier, const String& bundleIdentifier, const URL& pageURL)
{
    notImplemented();
}

void MediaUsageManager::removeMediaSession(WebCore::MediaSessionIdentifier identifier)
{
    notImplemented();
}

void MediaUsageManager::updateMediaUsage(WebCore::MediaSessionIdentifier identifier, const WebCore::MediaUsageInfo& mediaUsageInfo)
{
    notImplemented();
}

} // namespace WebKit

#endif // ENABLE(MEDIA_USAGE)
