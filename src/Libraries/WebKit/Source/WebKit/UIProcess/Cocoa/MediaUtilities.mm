/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#import "MediaUtilities.h"

namespace WebKit {

_WKMediaCaptureStateDeprecated toWKMediaCaptureStateDeprecated(WebCore::MediaProducerMediaStateFlags state)
{
    _WKMediaCaptureStateDeprecated mediaCaptureState = _WKMediaCaptureStateDeprecatedNone;
    if (state & WebCore::MediaProducerMediaState::HasActiveAudioCaptureDevice)
        mediaCaptureState |= _WKMediaCaptureStateDeprecatedActiveMicrophone;
    if (state & WebCore::MediaProducerMediaState::HasActiveVideoCaptureDevice)
        mediaCaptureState |= _WKMediaCaptureStateDeprecatedActiveCamera;
    if (state & WebCore::MediaProducerMediaState::HasMutedAudioCaptureDevice)
        mediaCaptureState |= _WKMediaCaptureStateDeprecatedMutedMicrophone;
    if (state & WebCore::MediaProducerMediaState::HasMutedVideoCaptureDevice)
        mediaCaptureState |= _WKMediaCaptureStateDeprecatedMutedCamera;

    return mediaCaptureState;
}

_WKMediaMutedState toWKMediaMutedState(WebCore::MediaProducerMutedStateFlags state)
{
    _WKMediaMutedState mediaMutedState = _WKMediaNoneMuted;
    if (state & WebCore::MediaProducerMutedState::AudioIsMuted)
        mediaMutedState |= _WKMediaAudioMuted;
    if (state & WebCore::MediaProducer::AudioAndVideoCaptureIsMuted)
        mediaMutedState |= _WKMediaCaptureDevicesMuted;
    if (state & WebCore::MediaProducerMutedState::ScreenCaptureIsMuted)
        mediaMutedState |= _WKMediaScreenCaptureMuted;
    return mediaMutedState;
}

} // namespace WebKit
