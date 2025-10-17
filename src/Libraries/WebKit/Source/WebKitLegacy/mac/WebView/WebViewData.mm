/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#import "WebViewData.h"

#import "WebKitLogging.h"
#import "WebPreferenceKeysPrivate.h"
#import "WebSelectionServiceController.h"
#import "WebViewGroup.h"
#import "WebViewInternal.h"
#import "WebViewRenderingUpdateScheduler.h"
#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/AlternativeTextUIController.h>
#import <WebCore/HistoryItem.h>
#import <WebCore/RunLoopObserver.h>
#import <WebCore/TextIndicatorWindow.h>
#import <WebCore/ValidationBubble.h>
#import <WebCore/WebCoreJITOperations.h>
#import <pal/spi/mac/NSWindowSPI.h>
#import <wtf/MainThread.h>
#import <wtf/RunLoop.h>
#import <wtf/SetForScope.h>

#if PLATFORM(IOS_FAMILY)
#import "WebGeolocationProviderIOS.h"
#endif

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)
#import "WebMediaPlaybackTargetPicker.h"
#endif

#if PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)
#import <WebCore/PlaybackSessionInterfaceMac.h>
#import <WebCore/PlaybackSessionModelMediaElement.h>
#endif

BOOL applicationIsTerminating = NO;
int pluginDatabaseClientCount = 0;

#if PLATFORM(MAC)

@implementation WebWindowVisibilityObserver

- (instancetype)initWithView:(WebView *)view
{
    self = [super init];
    if (!self)
        return nil;

    _view = view;
    return self;
}

- (void)startObserving:(NSWindow *)window
{
    // An NSView derived object such as WebView cannot observe these notifications, because NSView itself observes them.
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(_windowVisibilityChanged:) name:NSWindowDidOrderOffScreenNotification object:window];
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(_windowVisibilityChanged:) name:NSWindowDidOrderOnScreenNotification object:window];
}

- (void)stopObserving:(NSWindow *)window
{
    [[NSNotificationCenter defaultCenter] removeObserver:self name:NSWindowDidOrderOffScreenNotification object:window];
    [[NSNotificationCenter defaultCenter] removeObserver:self name:NSWindowDidOrderOnScreenNotification object:window];
}

- (void)_windowVisibilityChanged:(NSNotification *)notification
{
    [_view _windowVisibilityChanged:notification];
}

@end

#endif // PLATFORM(MAC)

@implementation WebViewPrivate

+ (void)initialize
{
#if !PLATFORM(IOS_FAMILY)
    JSC::initialize();
    WTF::initializeMainThread();
    WebCore::populateJITOperations();
#endif
}

- (id)init 
{
    self = [super init];
    if (!self)
        return nil;

    allowsUndo = YES;
    usesPageCache = YES;
    shouldUpdateWhileOffscreen = YES;

#if !PLATFORM(IOS_FAMILY)
    windowOcclusionDetectionEnabled = YES;
#endif

    zoomMultiplier = 1;
    zoomsTextOnly = NO;

#if PLATFORM(MAC)
    interactiveFormValidationEnabled = YES;
#else
    interactiveFormValidationEnabled = NO;
#endif
    // The default value should be synchronized with WebCore/page/Settings.cpp.
    validationMessageTimerMagnification = 50;

#if PLATFORM(IOS_FAMILY)
    isStopping = NO;
    _geolocationProvider = [WebGeolocationProviderIOS sharedGeolocationProvider];
#endif

    shouldCloseWithWindow = false;

    pluginDatabaseClientCount++;

    m_alternativeTextUIController = makeUnique<WebCore::AlternativeTextUIController>();

    return self;
}

- (void)dealloc
{    
    ASSERT(applicationIsTerminating || !page);
    ASSERT(applicationIsTerminating || !preferences);
#if !PLATFORM(IOS_FAMILY)
    ASSERT(!insertionPasteboard);
#endif
#if ENABLE(VIDEO)
    ASSERT(!fullscreenController);
#endif

    [super dealloc];
}

@end
