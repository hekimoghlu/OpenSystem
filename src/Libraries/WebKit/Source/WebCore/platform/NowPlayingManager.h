/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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

#include "NowPlayingInfo.h"
#include "PlatformMediaSession.h"
#include "RemoteCommandListener.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class NowPlayingManagerClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::NowPlayingManagerClient> : std::true_type { };
}

namespace WebCore {

class Image;
struct NowPlayingInfo;

class NowPlayingManagerClient : public CanMakeWeakPtr<NowPlayingManagerClient> {
public:
    virtual ~NowPlayingManagerClient() = default;
    virtual void didReceiveRemoteControlCommand(PlatformMediaSession::RemoteControlCommandType, const PlatformMediaSession::RemoteCommandArgument&) = 0;
};

class WEBCORE_EXPORT NowPlayingManager : public RemoteCommandListenerClient {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(NowPlayingManager, WEBCORE_EXPORT);
public:
    NowPlayingManager();
    ~NowPlayingManager();

    void didReceiveRemoteControlCommand(PlatformMediaSession::RemoteControlCommandType, const PlatformMediaSession::RemoteCommandArgument&) final;

    void addSupportedCommand(PlatformMediaSession::RemoteControlCommandType);
    void removeSupportedCommand(PlatformMediaSession::RemoteControlCommandType);
    RemoteCommandListener::RemoteCommandsSet supportedCommands() const;

    void addClient(NowPlayingManagerClient&);
    void removeClient(NowPlayingManagerClient&);

    void clearNowPlayingInfo();
    bool setNowPlayingInfo(const NowPlayingInfo&);
    void setSupportsSeeking(bool);
    void setSupportedRemoteCommands(const RemoteCommandListener::RemoteCommandsSet&);
    void updateSupportedCommands();

private:
    virtual void clearNowPlayingInfoPrivate();
    virtual void setNowPlayingInfoPrivate(const NowPlayingInfo&, bool shouldUpdateNowPlayingSuppression);
    void ensureRemoteCommandListenerCreated();
    RefPtr<RemoteCommandListener> m_remoteCommandListener;
    WeakPtr<NowPlayingManagerClient> m_client;
    std::optional<NowPlayingInfo> m_nowPlayingInfo;
    struct ArtworkCache {
        String src;
        RefPtr<Image> image;
    };
    std::optional<ArtworkCache> m_nowPlayingInfoArtwork;
    bool m_setAsNowPlayingApplication { false };
};

}
