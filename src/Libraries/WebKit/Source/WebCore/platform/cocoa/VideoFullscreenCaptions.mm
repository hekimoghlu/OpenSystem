/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#import "VideoFullscreenCaptions.h"

#import "FloatSize.h"
#import <QuartzCore/CALayer.h>
#import <QuartzCore/CATransaction.h>

namespace WebCore {

VideoFullscreenCaptions::VideoFullscreenCaptions() = default;
VideoFullscreenCaptions::~VideoFullscreenCaptions() = default;

void VideoFullscreenCaptions::setTrackRepresentationImage(PlatformImagePtr textTrack)
{
    if (m_captionsLayerHidden) {
        m_captionsLayerContents = (__bridge id)textTrack.get();
        return;
    }
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    [m_captionsLayer setContents:(__bridge id)textTrack.get()];
    [CATransaction commit];
}

void VideoFullscreenCaptions::setTrackRepresentationContentsScale(float scale)
{
    [m_captionsLayer setContentsScale:scale];
}

void VideoFullscreenCaptions::setTrackRepresentationHidden(bool hidden)
{
    if (hidden == m_captionsLayerHidden)
        return;
    m_captionsLayerHidden = hidden;

    // LinearMediaKit will un-hide the captionsLayer, so ensure the layer
    // is visually hidden by removing (and storing) the contents of the
    // captionsLayer.
    [m_captionsLayer setHidden:m_captionsLayerHidden];
    if (m_captionsLayerHidden) {
        m_captionsLayerContents = [m_captionsLayer contents];
        [m_captionsLayer setContents:nil];
    } else {
        [m_captionsLayer setContents:m_captionsLayerContents.get()];
        m_captionsLayerContents = nil;
    }
}

CALayer *VideoFullscreenCaptions::captionsLayer()
{
    if (!m_captionsLayer) {
        m_captionsLayer = adoptNS([[CALayer alloc] init]);
        [m_captionsLayer setName:@"Captions layer"];
    }
    return m_captionsLayer.get();
}

void VideoFullscreenCaptions::setCaptionsFrame(const CGRect& frame)
{
    [captionsLayer() setFrame:frame];
}

void VideoFullscreenCaptions::setupCaptionsLayer(CALayer *parent, const WebCore::FloatSize& initialSize)
{
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    [captionsLayer() removeFromSuperlayer];
    [parent addSublayer:captionsLayer()];
    captionsLayer().zPosition = FLT_MAX;
    [captionsLayer() setAnchorPoint:CGPointZero];
    [captionsLayer() setBounds:CGRectMake(0, 0, initialSize.width(), initialSize.height())];
    [CATransaction commit];
}

void VideoFullscreenCaptions::removeCaptionsLayer()
{
    [m_captionsLayer removeFromSuperlayer];
    m_captionsLayer = nullptr;
}

}
