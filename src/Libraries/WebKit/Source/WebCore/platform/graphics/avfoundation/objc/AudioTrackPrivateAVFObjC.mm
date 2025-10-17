/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#import "config.h"
#import "AudioTrackPrivateAVFObjC.h"
#import "AVTrackPrivateAVFObjCImpl.h"
#import "MediaSelectionGroupAVFObjC.h"
#import <wtf/TZoneMallocInlines.h>

#if ENABLE(VIDEO)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioTrackPrivateAVFObjC);

AudioTrackPrivateAVFObjC::AudioTrackPrivateAVFObjC(AVPlayerItemTrack* track)
    : AudioTrackPrivateAVFObjC(makeUnique<AVTrackPrivateAVFObjCImpl>(track))
{
}

AudioTrackPrivateAVFObjC::AudioTrackPrivateAVFObjC(AVAssetTrack* track)
    : AudioTrackPrivateAVFObjC(makeUnique<AVTrackPrivateAVFObjCImpl>(track))
{
}

AudioTrackPrivateAVFObjC::AudioTrackPrivateAVFObjC(MediaSelectionOptionAVFObjC& option)
    : AudioTrackPrivateAVFObjC(makeUnique<AVTrackPrivateAVFObjCImpl>(option))
{
}

AudioTrackPrivateAVFObjC::AudioTrackPrivateAVFObjC(std::unique_ptr<AVTrackPrivateAVFObjCImpl>&& impl)
    : m_impl(WTFMove(impl))
    , m_audioTrackConfigurationObserver([this] { audioTrackConfigurationChanged(); })
{
    m_impl->setAudioTrackConfigurationObserver(m_audioTrackConfigurationObserver);
    resetPropertiesFromTrack();
}

AudioTrackPrivateAVFObjC::~AudioTrackPrivateAVFObjC() = default;

void AudioTrackPrivateAVFObjC::resetPropertiesFromTrack()
{
    // Don't call this->setEnabled() because it also sets the enabled state of the
    // AVPlayerItemTrack
    AudioTrackPrivateAVF::setEnabled(m_impl->enabled());

    setTrackIndex(m_impl->index());
    setKind(m_impl->audioKind());
    setId(m_impl->id());
    setLabel(m_impl->label());
    setLanguage(m_impl->language());
    setConfiguration(m_impl->audioTrackConfiguration());
}

void AudioTrackPrivateAVFObjC::audioTrackConfigurationChanged()
{
    setConfiguration(m_impl->audioTrackConfiguration());
}

void AudioTrackPrivateAVFObjC::setPlayerItemTrack(AVPlayerItemTrack *track)
{
    m_impl = makeUnique<AVTrackPrivateAVFObjCImpl>(track);
    resetPropertiesFromTrack();
}

AVPlayerItemTrack* AudioTrackPrivateAVFObjC::playerItemTrack()
{
    return m_impl->playerItemTrack();
}

void AudioTrackPrivateAVFObjC::setAssetTrack(AVAssetTrack *track)
{
    m_impl = makeUnique<AVTrackPrivateAVFObjCImpl>(track);
    resetPropertiesFromTrack();
}

AVAssetTrack* AudioTrackPrivateAVFObjC::assetTrack()
{
    return m_impl->assetTrack();
}

void AudioTrackPrivateAVFObjC::setMediaSelectionOption(MediaSelectionOptionAVFObjC& option)
{
    m_impl = makeUnique<AVTrackPrivateAVFObjCImpl>(option);
    resetPropertiesFromTrack();
}

MediaSelectionOptionAVFObjC* AudioTrackPrivateAVFObjC::mediaSelectionOption()
{
    return m_impl->mediaSelectionOption();
}

void AudioTrackPrivateAVFObjC::setEnabled(bool enabled)
{
    AudioTrackPrivateAVF::setEnabled(enabled);
    m_impl->setEnabled(enabled);
}

}

#endif

