/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#import "WKAccessibilityWebPageObjectIOS.h"

#if PLATFORM(IOS_FAMILY)

#import "WebFrame.h"
#import "WebPage.h"
#import <WebCore/IntPoint.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/Page.h>
#import <WebCore/WAKAppKitStubs.h>

/* 
 The implementation of this class will be augmented by an accessibility bundle that is loaded only when accessibility is requested to be enabled.
 */

@implementation WKAccessibilityWebPageObject

- (instancetype)init
{
    self = [super init];
    if (!self)
        return nil;
    
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(_accessibilityCategoryInstalled:) name:@"AccessibilityCategoryInstalled" object:nil];

    return self;
}

- (void)dealloc
{
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    [_remoteTokenData release];
    [super dealloc];
}

- (void)_accessibilityCategoryInstalled:(id)notification
{
    // Accessibility bundle will override this method so that it knows when to initialize the accessibility runtime within the WebProcess.
}

- (double)pageScale
{
    return m_page->pageScaleFactor();
}

- (id)accessibilityHitTest:(NSPoint)point
{
    if (!m_page)
        return nil;

    WebCore::IntPoint convertedPoint = m_page->accessibilityScreenToRootView(WebCore::IntPoint(point));

    // If we are hit-testing a remote element, offset the hit test by the scroll of the web page.
    if (RefPtr remoteLocalFrame = [self remoteLocalFrame]) {
        if (CheckedPtr frameView = remoteLocalFrame->view())
            convertedPoint.moveBy(frameView->scrollPosition());
    }

    return [[self accessibilityRootObjectWrapper] accessibilityHitTest:convertedPoint];
}

@end

#endif // PLATFORM(IOS_FAMILY)

