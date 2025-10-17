/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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
#import "WebStringTruncator.h"

#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/FontCascade.h>
#import <WebCore/FontPlatformData.h>
#import <WebCore/StringTruncator.h>
#import <WebCore/WebCoreJITOperations.h>
#import <wtf/MainThread.h>
#import <wtf/NeverDestroyed.h>

static WebCore::FontCascade& fontFromNSFont(NSFont *font)
{
    static NeverDestroyed<RetainPtr<NSFont>> currentNSFont;
    static NeverDestroyed<WebCore::FontCascade> currentFont;
    if ([font isEqual:currentNSFont.get().get()])
        return currentFont;
    currentNSFont.get() = font;
    currentFont.get() = WebCore::FontCascade(WebCore::FontPlatformData((__bridge CTFontRef)font, [font pointSize]));
    return currentFont;
}

@implementation WebStringTruncator

+ (void)initialize
{
    JSC::initialize();
    WTF::initializeMainThread();
    WebCore::populateJITOperations();
}

+ (NSString *)centerTruncateString:(NSString *)string toWidth:(float)maxWidth
{
    static NeverDestroyed<RetainPtr<NSFont>> menuFont = [NSFont menuFontOfSize:0];

    ASSERT(menuFont.get());
    if (!menuFont.get())
        return nil;

    return WebCore::StringTruncator::centerTruncate(string, maxWidth, fontFromNSFont(menuFont.get().get()));
}

+ (NSString *)centerTruncateString:(NSString *)string toWidth:(float)maxWidth withFont:(NSFont *)font
{
    if (!font)
        return nil;

    return WebCore::StringTruncator::centerTruncate(string, maxWidth, fontFromNSFont(font));
}

+ (NSString *)rightTruncateString:(NSString *)string toWidth:(float)maxWidth withFont:(NSFont *)font
{
    if (!font)
        return nil;

    return WebCore::StringTruncator::rightTruncate(string, maxWidth, fontFromNSFont(font));
}

+ (float)widthOfString:(NSString *)string font:(NSFont *)font
{
    if (!font)
        return 0;

    return WebCore::StringTruncator::width(string, fontFromNSFont(font));
}

@end
