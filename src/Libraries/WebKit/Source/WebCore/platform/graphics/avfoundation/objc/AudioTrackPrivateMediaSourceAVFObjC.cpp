/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#include "AudioTrackPrivateMediaSourceAVFObjC.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MEDIA_SOURCE)

#include "AVTrackPrivateAVFObjCImpl.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioTrackPrivateMediaSourceAVFObjC);

AudioTrackPrivateMediaSourceAVFObjC::AudioTrackPrivateMediaSourceAVFObjC(AVAssetTrack* track)
    : m_impl(makeUnique<AVTrackPrivateAVFObjCImpl>(track))
{
    resetPropertiesFromTrack();
}

AudioTrackPrivateMediaSourceAVFObjC::~AudioTrackPrivateMediaSourceAVFObjC() = default;

void AudioTrackPrivateMediaSourceAVFObjC::resetPropertiesFromTrack()
{
    setKind(m_impl->audioKind());
    setId(m_impl->id());
    setLabel(m_impl->label());
    setLanguage(m_impl->language());
}

void AudioTrackPrivateMediaSourceAVFObjC::setAssetTrack(AVAssetTrack *track)
{
    m_impl = makeUnique<AVTrackPrivateAVFObjCImpl>(track);
    resetPropertiesFromTrack();
}

AVAssetTrack* AudioTrackPrivateMediaSourceAVFObjC::assetTrack()
{
    return m_impl->assetTrack();
}

void AudioTrackPrivateMediaSourceAVFObjC::setEnabled(bool enabled)
{
    if (enabled == this->enabled())
        return;

    AudioTrackPrivateAVF::setEnabled(enabled);
}

}

#endif
