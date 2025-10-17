/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
#import "WAKClipView.h"

#if PLATFORM(IOS_FAMILY)

#import "WAKViewInternal.h"
#import <wtf/Assertions.h>

@implementation WAKClipView

@synthesize documentView = _documentView;
@synthesize copiesOnScroll = _copiesOnScroll;

- (id)initWithFrame:(CGRect)rect
{
    WKViewRef view = WKViewCreateWithFrame(rect, &viewContext);
    self = [self _initWithViewRef:view];
    WAKRelease(view);
    return self;
}

- (void)dealloc
{
    [_documentView release];
    [super dealloc];
}

// WAK internal function for WAKScrollView.
- (void)_setDocumentView:(WAKView *)aView
{
    if (_documentView == aView)
        return;

    [_documentView removeFromSuperview];
    [_documentView release];
    _documentView = [aView retain];
    [self addSubview:_documentView];
}

- (CGRect)documentVisibleRect 
{     
    if (_documentView)
        return WKViewConvertRectFromSuperview([_documentView _viewRef], [self bounds]);
    return CGRectZero;
}

@end

#endif // PLATFORM(IOS_FAMILY)
