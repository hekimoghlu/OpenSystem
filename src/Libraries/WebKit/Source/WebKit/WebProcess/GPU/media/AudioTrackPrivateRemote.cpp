/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#include "AudioTrackPrivateRemote.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "AudioTrackPrivateRemoteConfiguration.h"
#include "GPUProcessConnection.h"
#include "MediaPlayerPrivateRemote.h"
#include "RemoteMediaPlayerProxyMessages.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioTrackPrivateRemote);

AudioTrackPrivateRemote::AudioTrackPrivateRemote(GPUProcessConnection& gpuProcessConnection, WebCore::MediaPlayerIdentifier playerIdentifier, AudioTrackPrivateRemoteConfiguration&& configuration)
    : m_gpuProcessConnection(gpuProcessConnection)
    , m_id(configuration.trackId)
    , m_playerIdentifier(playerIdentifier)
{
    updateConfiguration(WTFMove(configuration));
}

void AudioTrackPrivateRemote::setEnabled(bool enabled)
{
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    if (!gpuProcessConnection)
        return;

    if (enabled != this->enabled())
        gpuProcessConnection->connection().send(Messages::RemoteMediaPlayerProxy::AudioTrackSetEnabled(m_id, enabled), m_playerIdentifier);

    AudioTrackPrivate::setEnabled(enabled);
}

void AudioTrackPrivateRemote::updateConfiguration(AudioTrackPrivateRemoteConfiguration&& configuration)
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
        if (changed) {
            notifyClients([label = crossThreadCopy(m_label)](auto& client) {
                client.labelChanged(AtomString { label });
            });
        };
    }

    if (configuration.language != m_language) {
        auto changed = !m_language.isEmpty();
        m_language = configuration.language;
        if (changed) {
            notifyClients([language = crossThreadCopy(m_language)](auto& client) {
                client.languageChanged(AtomString { language });
            });
        };
    }

    m_trackIndex = configuration.trackIndex;
    m_startTimeVariance = configuration.startTimeVariance;
    m_kind = configuration.kind;
    setConfiguration(WTFMove(configuration.trackConfiguration));
    
    AudioTrackPrivate::setEnabled(configuration.enabled);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
