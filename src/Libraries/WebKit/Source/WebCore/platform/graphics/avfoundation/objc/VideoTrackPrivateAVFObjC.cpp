/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#import "VideoTrackPrivateAVFObjC.h"

#if ENABLE(VIDEO)

#import "AVTrackPrivateAVFObjCImpl.h"
#import "MediaSelectionGroupAVFObjC.h"
#import "PlatformVideoTrackConfiguration.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VideoTrackPrivateAVFObjC);

VideoTrackPrivateAVFObjC::VideoTrackPrivateAVFObjC(AVPlayerItemTrack* track)
    : VideoTrackPrivateAVFObjC(makeUnique<AVTrackPrivateAVFObjCImpl>(track))
{
}

VideoTrackPrivateAVFObjC::VideoTrackPrivateAVFObjC(AVAssetTrack* track)
    : VideoTrackPrivateAVFObjC(makeUnique<AVTrackPrivateAVFObjCImpl>(track))
{
}

VideoTrackPrivateAVFObjC::VideoTrackPrivateAVFObjC(MediaSelectionOptionAVFObjC& option)
    : VideoTrackPrivateAVFObjC(makeUnique<AVTrackPrivateAVFObjCImpl>(option))
{
}

VideoTrackPrivateAVFObjC::VideoTrackPrivateAVFObjC(std::unique_ptr<AVTrackPrivateAVFObjCImpl>&& impl)
    : m_impl(WTFMove(impl))
    , m_videoTrackConfigurationObserver([this] { videoTrackConfigurationChanged(); })
{
    m_impl->setVideoTrackConfigurationObserver(m_videoTrackConfigurationObserver);
    resetPropertiesFromTrack();
}

void VideoTrackPrivateAVFObjC::resetPropertiesFromTrack()
{
    // Don't call this->setSelected() because it also sets the enabled state of the
    // AVPlayerItemTrack
    VideoTrackPrivateAVF::setSelected(m_impl->enabled());

    setTrackIndex(m_impl->id());
    setKind(m_impl->videoKind());
    setId(m_impl->id());
    setLabel(m_impl->label());
    setLanguage(m_impl->language());
    setConfiguration(m_impl->videoTrackConfiguration());
}

void VideoTrackPrivateAVFObjC::videoTrackConfigurationChanged()
{
    setConfiguration(m_impl->videoTrackConfiguration());
}

void VideoTrackPrivateAVFObjC::setPlayerItemTrack(AVPlayerItemTrack *track)
{
    m_impl = makeUnique<AVTrackPrivateAVFObjCImpl>(track);
    resetPropertiesFromTrack();
}

AVPlayerItemTrack* VideoTrackPrivateAVFObjC::playerItemTrack()
{
    return m_impl->playerItemTrack();
}

void VideoTrackPrivateAVFObjC::setAssetTrack(AVAssetTrack *track)
{
    m_impl = makeUnique<AVTrackPrivateAVFObjCImpl>(track);
    resetPropertiesFromTrack();
}

AVAssetTrack* VideoTrackPrivateAVFObjC::assetTrack()
{
    return m_impl->assetTrack();
}

void VideoTrackPrivateAVFObjC::setMediaSelectonOption(MediaSelectionOptionAVFObjC& option)
{
    m_impl = makeUnique<AVTrackPrivateAVFObjCImpl>(option);
    resetPropertiesFromTrack();
}

MediaSelectionOptionAVFObjC* VideoTrackPrivateAVFObjC::mediaSelectionOption()
{
    return m_impl->mediaSelectionOption();
}

void VideoTrackPrivateAVFObjC::setSelected(bool enabled)
{
    VideoTrackPrivateAVF::setSelected(enabled);
    m_impl->setEnabled(enabled);
}
    
}

#endif
