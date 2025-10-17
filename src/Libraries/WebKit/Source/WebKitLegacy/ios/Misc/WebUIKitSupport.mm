/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#if PLATFORM(IOS_FAMILY)

#import "WebUIKitSupport.h"

#import "WebDatabaseManagerInternal.h"
#import "WebLocalizableStringsInternal.h"
#import "WebPlatformStrategies.h"
#import "WebPreferencesDefinitions.h"
#import "WebViewPrivate.h"
#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/BreakLines.h>
#import <WebCore/PathUtilities.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/Settings.h>
#import <WebCore/WebBackgroundTaskController.h>
#import <WebCore/WebCoreThreadSystemInterface.h>
#import <wtf/spi/darwin/dyldSPI.h>

using namespace WebCore;

// See <rdar://problem/7902473> Optimize WebLocalizedString for why we do this on a background thread on a timer callback
static void LoadWebLocalizedStringsTimerCallback(CFRunLoopTimerRef timer, void *info)
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0), ^ {
        // We don't care if we find this string, but searching for it will load the plist and save the results.
        // FIXME: It would be nicer to do this in a more direct way.
        UI_STRING_KEY_INTERNAL("Typing", "Typing (Undo action name)", "Undo action name");
    });
}

static void LoadWebLocalizedStrings()
{
    auto timer = adoptCF(CFRunLoopTimerCreate(kCFAllocatorDefault, CFAbsoluteTimeGetCurrent(), 0, 0, 0, &LoadWebLocalizedStringsTimerCallback, NULL));
    CFRunLoopAddTimer(CFRunLoopGetCurrent(), timer.get(), kCFRunLoopCommonModes);
}

void WebKitInitialize(void)
{
    static bool webkitInitialized;
    if (webkitInitialized)
        return;

    ASSERT(pthread_main_np());
    webkitInitialized = true;
    InitWebCoreThreadSystemInterface();
    [WebView enableWebThread];

    // Initialize our platform strategies.
    WebPlatformStrategies::initializeIfNecessary();

    // We'd rather eat this cost at startup than slow down situations that need to be responsive.
    // See <rdar://problem/6776301>.
    LoadWebLocalizedStrings();
    
    // This needs to be called before any requests are made in the process, <rdar://problem/9691871>
    WebCore::initializeHTTPConnectionSettingsOnStartup();
}

float WebKitGetMinimumZoomFontSize(void)
{
    return DEFAULT_VALUE_FOR_MinimumZoomFontSize;
}

int WebKitGetLastLineBreakInBuffer(UChar *characters, int position, int length)
{
    unsigned lastBreakPos = position;
    unsigned breakPos = 0;
    CachedLineBreakIteratorFactory lineBreakIteratorFactory(StringView { std::span(characters, length) });
    while (static_cast<int>(breakPos = BreakLines::nextBreakablePosition(lineBreakIteratorFactory, breakPos)) < position)
        lastBreakPos = breakPos++;
    return static_cast<int>(lastBreakPos) < position ? lastBreakPos : INT_MAX;
}

const char *WebKitPlatformSystemRootDirectory(void)
{
#if PLATFORM(IOS_FAMILY_SIMULATOR)
    static const char *platformSystemRootDirectory = nil;
    if (!platformSystemRootDirectory) {
        char *simulatorRoot = getenv("IPHONE_SIMULATOR_ROOT");
        platformSystemRootDirectory = simulatorRoot ? simulatorRoot : "/";
    }
    return platformSystemRootDirectory;
#else
    return "/";
#endif
}

void WebKitSetBackgroundAndForegroundNotificationNames(NSString *didEnterBackgroundName, NSString *willEnterForegroundName)
{
    // FIXME: Remove this function.
}

void WebKitSetInvalidWebBackgroundTaskIdentifier(WebBackgroundTaskIdentifier taskIdentifier)
{
    [[WebBackgroundTaskController sharedController] setInvalidBackgroundTaskIdentifier:taskIdentifier];
}

void WebKitSetStartBackgroundTaskBlock(StartBackgroundTaskBlock startBlock)
{
    [[WebBackgroundTaskController sharedController] setBackgroundTaskStartBlock:startBlock];
}

void WebKitSetEndBackgroundTaskBlock(EndBackgroundTaskBlock endBlock)
{
    [[WebBackgroundTaskController sharedController] setBackgroundTaskEndBlock:endBlock];
}

CGPathRef WebKitCreatePathWithShrinkWrappedRects(NSArray* cgRects, CGFloat radius)
{
    Vector<FloatRect> rects;
    rects.reserveInitialCapacity([cgRects count]);

    const char* cgRectEncodedString = @encode(CGRect);

    for (NSValue *rectValue in cgRects) {
        CGRect cgRect;
        [rectValue getValue:&cgRect];

        if (strcmp(cgRectEncodedString, rectValue.objCType))
            return nullptr;
        rects.append(cgRect);
    }

    return CGPathRetain(PathUtilities::pathWithShrinkWrappedRects(rects, radius).platformPath());
}

#endif // PLATFORM(IOS_FAMILY)
