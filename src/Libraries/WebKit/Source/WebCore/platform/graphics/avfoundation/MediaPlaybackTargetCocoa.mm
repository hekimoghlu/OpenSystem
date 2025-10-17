/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
#import "MediaPlaybackTargetCocoa.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET)

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

MediaPlaybackTargetContextCocoa::MediaPlaybackTargetContextCocoa(RetainPtr<AVOutputContext>&& outputContext)
    : MediaPlaybackTargetContext(Type::AVOutputContext)
    , m_outputContext(WTFMove(outputContext))
{
    ASSERT(m_outputContext);
}

MediaPlaybackTargetContextCocoa::~MediaPlaybackTargetContextCocoa() = default;

RetainPtr<AVOutputContext> MediaPlaybackTargetContextCocoa::outputContext() const
{
    return m_outputContext;
}

String MediaPlaybackTargetContextCocoa::deviceName() const
{
    if (![m_outputContext supportsMultipleOutputDevices])
        return [m_outputContext deviceName];

    auto outputDeviceNames = adoptNS([[NSMutableArray alloc] init]);
    for (AVOutputDevice *outputDevice in [m_outputContext outputDevices])
        [outputDeviceNames addObject:[outputDevice deviceName]];

    return [outputDeviceNames componentsJoinedByString:@" + "];
}

bool MediaPlaybackTargetContextCocoa::hasActiveRoute() const
{
    if ([m_outputContext respondsToSelector:@selector(supportsMultipleOutputDevices)] && [m_outputContext supportsMultipleOutputDevices] && [m_outputContext respondsToSelector:@selector(outputDevices)]) {
        for (AVOutputDevice *outputDevice in [m_outputContext outputDevices]) {
            if (outputDevice.deviceFeatures & (AVOutputDeviceFeatureVideo | AVOutputDeviceFeatureAudio))
                return true;
        }
    } else if ([m_outputContext respondsToSelector:@selector(outputDevice)]) {
        if (auto *outputDevice = [m_outputContext outputDevice])
            return outputDevice.deviceFeatures & (AVOutputDeviceFeatureVideo | AVOutputDeviceFeatureAudio);
    }
    return m_outputContext.get().deviceName;
}
bool MediaPlaybackTargetContextCocoa::supportsRemoteVideoPlayback() const
{
    if (![m_outputContext respondsToSelector:@selector(supportsMultipleOutputDevices)] || ![m_outputContext supportsMultipleOutputDevices] || ![m_outputContext respondsToSelector:@selector(outputDevices)]) {
        if (auto *outputDevice = [m_outputContext outputDevice]) {
            if (outputDevice.deviceFeatures & AVOutputDeviceFeatureVideo)
                return true;
        }
        return false;
    }

    for (AVOutputDevice *outputDevice in [m_outputContext outputDevices]) {
        if (outputDevice.deviceFeatures & AVOutputDeviceFeatureVideo)
            return true;
    }

    return false;
}

Ref<MediaPlaybackTarget> MediaPlaybackTargetCocoa::create(MediaPlaybackTargetContextCocoa&& context)
{
    return adoptRef(*new MediaPlaybackTargetCocoa(WTFMove(context)));
}

MediaPlaybackTargetCocoa::MediaPlaybackTargetCocoa(MediaPlaybackTargetContextCocoa&& context)
    : m_context(context.outputContext())
{
}

#if PLATFORM(IOS_FAMILY) && !PLATFORM(IOS_FAMILY_SIMULATOR) && !PLATFORM(MACCATALYST)
Ref<MediaPlaybackTargetCocoa> MediaPlaybackTargetCocoa::create()
{
    auto *routingContextUID = [[PAL::getAVAudioSessionClass() sharedInstance] routingContextUID];
    return adoptRef(*new MediaPlaybackTargetCocoa(MediaPlaybackTargetContextCocoa([PAL::getAVOutputContextClass() outputContextForID:routingContextUID])));
}
#endif

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET)
