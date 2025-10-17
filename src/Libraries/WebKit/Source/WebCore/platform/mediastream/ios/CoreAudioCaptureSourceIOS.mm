/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#import "CoreAudioCaptureSourceIOS.h"

#if ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)

#import "AVAudioSessionCaptureDeviceManager.h"
#import "Logging.h"
#import <AVFoundation/AVAudioSession.h>
#import <wtf/MainThread.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

using namespace WebCore;

@interface WebCoreAudioCaptureSourceIOSListener : NSObject {
    CoreAudioCaptureSourceFactoryIOS* _callback;
}

- (void)invalidate;
- (void)sessionMediaServicesWereReset:(NSNotification*)notification;
@end

@implementation WebCoreAudioCaptureSourceIOSListener
- (id)initWithCallback:(CoreAudioCaptureSourceFactoryIOS*)callback
{
    self = [super init];
    if (!self)
        return nil;

    _callback = callback;

    NSNotificationCenter* center = [NSNotificationCenter defaultCenter];
    AVAudioSession* session = [PAL::getAVAudioSessionClass() sharedInstance];

    [center addObserver:self selector:@selector(sessionMediaServicesWereReset:) name:AVAudioSessionMediaServicesWereResetNotification object:session];

    return self;
}

- (void)invalidate
{
    _callback = nullptr;
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)sessionMediaServicesWereReset:(NSNotification*)notification
{
    UNUSED_PARAM(notification);
    ASSERT(_callback);

    if (!_callback)
        return;

    // mediaserverd crashed and was relaunched, rebuild everything. See
    // https://developer.apple.com/library/content/qa/qa1749/_index.html.
    _callback->scheduleReconfiguration();
}
@end

namespace WebCore {

CoreAudioCaptureSourceFactoryIOS::CoreAudioCaptureSourceFactoryIOS()
    : m_listener(adoptNS([[WebCoreAudioCaptureSourceIOSListener alloc] initWithCallback:this]))
{
}

CoreAudioCaptureSourceFactoryIOS::~CoreAudioCaptureSourceFactoryIOS()
{
    [m_listener invalidate];
    m_listener = nullptr;
}

CoreAudioCaptureSourceFactory& CoreAudioCaptureSourceFactory::singleton()
{
    static NeverDestroyed<CoreAudioCaptureSourceFactoryIOS> factory;
    return factory.get();
}

}

#endif // ENABLE(MEDIA_STREAM) && PLATFORM(IOS_FAMILY)
