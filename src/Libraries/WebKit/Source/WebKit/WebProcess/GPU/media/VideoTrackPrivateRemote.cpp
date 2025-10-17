/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#include "VideoTrackPrivateRemote.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "GPUProcessConnection.h"
#include "MediaPlayerPrivateRemote.h"
#include "RemoteMediaPlayerProxyMessages.h"
#include "VideoTrackPrivateRemoteConfiguration.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoTrackPrivateRemote);

VideoTrackPrivateRemote::VideoTrackPrivateRemote(GPUProcessConnection& gpuProcessConnection, WebCore::MediaPlayerIdentifier playerIdentifier, VideoTrackPrivateRemoteConfiguration&& configuration)
    : m_gpuProcessConnection(gpuProcessConnection)
    , m_id(configuration.trackId)
    , m_playerIdentifier(playerIdentifier)
{
    updateConfiguration(WTFMove(configuration));
}

void VideoTrackPrivateRemote::setSelected(bool selected)
{
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    if (!gpuProcessConnection)
        return;

    if (selected != this->selected())
        gpuProcessConnection->connection().send(Messages::RemoteMediaPlayerProxy::VideoTrackSetSelected(m_id, selected), m_playerIdentifier);

    VideoTrackPrivate::setSelected(selected);
}

void VideoTrackPrivateRemote::updateConfiguration(VideoTrackPrivateRemoteConfiguration&& configuration)
{
    if (configuration.trackId != m_id) {
        m_id = configuration.trackId;
        notifyClients([id = m_id](auto& client) {
            client.idChanged(id);
        });
    }

    if (configuration.label != m_label) {
        auto changed = !m_label.isEmpty();
        m_label = configuration.label;
        if (changed && hasClients()) {
            notifyClients([label = crossThreadCopy(m_label)](auto& client) {
                client.labelChanged(AtomString { label });
            });
        }
    }

    if (configuration.language != m_language) {
        auto changed = !m_language.isEmpty();
        m_language = configuration.language;
        if (changed && hasClients()) {
            notifyClients([language = crossThreadCopy(m_language)](auto& client) {
                client.languageChanged(AtomString { language });
            });
        }
    }

    m_trackIndex = configuration.trackIndex;
    m_startTimeVariance = configuration.startTimeVariance;
    m_kind = configuration.kind;
    setConfiguration(WTFMove(configuration.trackConfiguration));
    VideoTrackPrivate::setSelected(configuration.selected);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
