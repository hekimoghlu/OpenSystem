/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "NowPlayingManager.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include "MediaSessionManagerCocoa.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NowPlayingManager);

NowPlayingManager::NowPlayingManager() = default;
NowPlayingManager::~NowPlayingManager() = default;

void NowPlayingManager::didReceiveRemoteControlCommand(PlatformMediaSession::RemoteControlCommandType type, const PlatformMediaSession::RemoteCommandArgument& argument)
{
    if (m_client)
        m_client->didReceiveRemoteControlCommand(type, argument);
}

void NowPlayingManager::addClient(NowPlayingManagerClient& client)
{
    m_client = client;
    ensureRemoteCommandListenerCreated();
}

void NowPlayingManager::removeClient(NowPlayingManagerClient& client)
{
    if (m_client.get() != &client)
        return;

    m_remoteCommandListener = nullptr;
    m_client.clear();
    m_nowPlayingInfo = { };

    clearNowPlayingInfo();
}

void NowPlayingManager::clearNowPlayingInfo()
{
    clearNowPlayingInfoPrivate();
    m_setAsNowPlayingApplication = false;
}

void NowPlayingManager::clearNowPlayingInfoPrivate()
{
#if PLATFORM(COCOA)
    MediaSessionManagerCocoa::clearNowPlayingInfo();
#endif
}

bool NowPlayingManager::setNowPlayingInfo(const NowPlayingInfo& nowPlayingInfo)
{
    if (m_nowPlayingInfo && *m_nowPlayingInfo == nowPlayingInfo)
        return false;

    bool shouldUpdateNowPlayingSuppression = [&] {
#if USE(NOW_PLAYING_ACTIVITY_SUPPRESSION)
        if (!m_nowPlayingInfo)
            return true;

        if (m_nowPlayingInfo->isVideo != nowPlayingInfo.isVideo)
            return true;

        if (m_nowPlayingInfo->metadata.sourceApplicationIdentifier != nowPlayingInfo.metadata.sourceApplicationIdentifier)
            return true;
#endif

        return false;
    }();

    m_nowPlayingInfo = nowPlayingInfo;

    // We do not want to send the artwork's image over each time nowPlayingInfo gets updated.
    // So if present we store it once locally. On the receiving end, a null imageData indicates to use the cached image.
    if (!nowPlayingInfo.metadata.artwork)
        m_nowPlayingInfoArtwork = { };
    else if (!m_nowPlayingInfoArtwork || nowPlayingInfo.metadata.artwork->src != m_nowPlayingInfoArtwork->src)
        m_nowPlayingInfoArtwork = ArtworkCache { nowPlayingInfo.metadata.artwork->src, nowPlayingInfo.metadata.artwork->image };
    else
        m_nowPlayingInfo->metadata.artwork->image = nullptr;

    setNowPlayingInfoPrivate(*m_nowPlayingInfo, shouldUpdateNowPlayingSuppression);
    m_setAsNowPlayingApplication = true;
    return true;
}

void NowPlayingManager::setNowPlayingInfoPrivate(const NowPlayingInfo& nowPlayingInfo, bool shouldUpdateNowPlayingSuppression)
{
    setSupportsSeeking(nowPlayingInfo.supportsSeeking);
#if PLATFORM(COCOA)
    if (nowPlayingInfo.metadata.artwork && !nowPlayingInfo.metadata.artwork->image) {
        ASSERT(m_nowPlayingInfoArtwork, "cached value must have been initialized");
        NowPlayingInfo nowPlayingInfoRebuilt = nowPlayingInfo;
        nowPlayingInfoRebuilt.metadata.artwork->image = m_nowPlayingInfoArtwork->image;
        MediaSessionManagerCocoa::setNowPlayingInfo(!m_setAsNowPlayingApplication, shouldUpdateNowPlayingSuppression, nowPlayingInfoRebuilt);
        return;
    }
    MediaSessionManagerCocoa::setNowPlayingInfo(!m_setAsNowPlayingApplication, shouldUpdateNowPlayingSuppression, nowPlayingInfo);
#else
    UNUSED_PARAM(shouldUpdateNowPlayingSuppression);
#endif
}

void NowPlayingManager::setSupportsSeeking(bool supports)
{
    if (m_remoteCommandListener)
        m_remoteCommandListener->setSupportsSeeking(supports);
}

void NowPlayingManager::addSupportedCommand(PlatformMediaSession::RemoteControlCommandType command)
{
    if (m_remoteCommandListener)
        m_remoteCommandListener->addSupportedCommand(command);
}

void NowPlayingManager::removeSupportedCommand(PlatformMediaSession::RemoteControlCommandType command)
{
    if (m_remoteCommandListener)
        m_remoteCommandListener->removeSupportedCommand(command);
}

RemoteCommandListener::RemoteCommandsSet NowPlayingManager::supportedCommands() const
{
    if (!m_remoteCommandListener)
        return { };
    return m_remoteCommandListener->supportedCommands();
}

void NowPlayingManager::setSupportedRemoteCommands(const RemoteCommandListener::RemoteCommandsSet& commands)
{
    if (m_remoteCommandListener)
        m_remoteCommandListener->setSupportedCommands(commands);
}

void NowPlayingManager::updateSupportedCommands()
{
    if (m_remoteCommandListener)
        m_remoteCommandListener->updateSupportedCommands();
}

void NowPlayingManager::ensureRemoteCommandListenerCreated()
{
    if (!m_remoteCommandListener)
        m_remoteCommandListener = RemoteCommandListener::create(*this);
}

}
