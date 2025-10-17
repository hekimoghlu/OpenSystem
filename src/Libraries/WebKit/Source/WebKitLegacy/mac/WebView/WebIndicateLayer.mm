/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#if PLATFORM(IOS_FAMILY) && ENABLE(REMOTE_INSPECTOR)

#import "WebIndicateLayer.h"

#import "WebFramePrivate.h"
#import "WebView.h"
#import <WebCore/ColorMac.h>
#import <WebCore/WAKWindow.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/NeverDestroyed.h>

using namespace WebCore;

@implementation WebIndicateLayer

- (id)initWithWebView:(WebView *)webView
{
    self = [super init];
    if (!self)
        return nil;

    _webView = webView;

    self.canDrawConcurrently = NO;
    self.contentsScale = [[_webView window] screenScale];

    // Blue highlight color.
    constexpr auto highlightColor = SRGBA<uint8_t> { 111, 168, 220, 168 };
    self.backgroundColor = cachedCGColor(highlightColor).get();

    return self;
}

- (void)layoutSublayers
{
    CGFloat documentScale = [[[_webView mainFrame] documentView] scale];
    [self setTransform:CATransform3DMakeScale(documentScale, documentScale, 1.0)];
    [self setFrame:[_webView frame]];
}

- (id<CAAction>)actionForKey:(NSString *)key
{
    // Disable all implicit animations.
    return nil;
}

@end

#endif
