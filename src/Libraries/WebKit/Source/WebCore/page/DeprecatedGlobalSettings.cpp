/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#include "DeprecatedGlobalSettings.h"

#include "AudioSession.h"
#include "HTMLMediaElement.h"
#include "MediaPlayer.h"
#include "MediaStrategy.h"
#include "PlatformMediaSessionManager.h"
#include "PlatformStrategies.h"
#include <wtf/NeverDestroyed.h>

#if PLATFORM(COCOA)
#include "MediaSessionManagerCocoa.h"
#endif

namespace WebCore {

DeprecatedGlobalSettings& DeprecatedGlobalSettings::shared()
{
    static NeverDestroyed<DeprecatedGlobalSettings> deprecatedGlobalSettings;
    return deprecatedGlobalSettings;
}

#if USE(AVFOUNDATION)
void DeprecatedGlobalSettings::setAVFoundationEnabled(bool enabled)
{
    if (shared().m_AVFoundationEnabled == enabled)
        return;

    shared().m_AVFoundationEnabled = enabled;
    platformStrategies()->mediaStrategy().resetMediaEngines();
}
#endif

#if USE(GSTREAMER)
void DeprecatedGlobalSettings::setGStreamerEnabled(bool enabled)
{
    if (shared().m_GStreamerEnabled == enabled)
        return;

    shared().m_GStreamerEnabled = enabled;

#if ENABLE(VIDEO)
    platformStrategies()->mediaStrategy().resetMediaEngines();
#endif
}
#endif

// It's very important that this setting doesn't change in the middle of a document's lifetime.
// The Mac port uses this flag when registering and deregistering platform-dependent scrollbar
// objects. Therefore, if this changes at an unexpected time, deregistration may not happen
// correctly, which may cause the platform to follow dangling pointers.
void DeprecatedGlobalSettings::setMockScrollbarsEnabled(bool flag)
{
    shared().m_mockScrollbarsEnabled = flag;
    // FIXME: This should update scroll bars in existing pages.
}

void DeprecatedGlobalSettings::setUsesOverlayScrollbars(bool flag)
{
    shared().m_usesOverlayScrollbars = flag;
    // FIXME: This should update scroll bars in existing pages.
}

void DeprecatedGlobalSettings::setTrackingPreventionEnabled(bool flag)
{
    shared().m_trackingPreventionEnabled = flag;
}

#if PLATFORM(IOS_FAMILY)
void DeprecatedGlobalSettings::setAudioSessionCategoryOverride(unsigned sessionCategory)
{
    AudioSession::sharedSession().setCategoryOverride(static_cast<AudioSession::CategoryType>(sessionCategory));
}

unsigned DeprecatedGlobalSettings::audioSessionCategoryOverride()
{
    return static_cast<unsigned>(AudioSession::sharedSession().categoryOverride());
}

void DeprecatedGlobalSettings::setNetworkInterfaceName(const String& networkInterfaceName)
{
    shared().m_networkInterfaceName = networkInterfaceName;
}
#endif

#if USE(AUDIO_SESSION)
void DeprecatedGlobalSettings::setShouldManageAudioSessionCategory(bool flag)
{
    AudioSession::setShouldManageAudioSessionCategory(flag);
}

bool DeprecatedGlobalSettings::shouldManageAudioSessionCategory()
{
    return AudioSession::shouldManageAudioSessionCategory();
}
#endif

void DeprecatedGlobalSettings::setAllowsAnySSLCertificate(bool allowAnySSLCertificate)
{
    shared().m_allowsAnySSLCertificate = allowAnySSLCertificate;
}

bool DeprecatedGlobalSettings::allowsAnySSLCertificate()
{
    return shared().m_allowsAnySSLCertificate;
}

#if ENABLE(WEB_PUSH_NOTIFICATIONS)

bool DeprecatedGlobalSettings::builtInNotificationsEnabled()
{
    return shared().m_builtInNotificationsEnabled;
}

#endif

} // namespace WebCore
