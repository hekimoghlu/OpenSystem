/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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

#if USE(APPKIT)

#import "WebCoreView.h"

@interface NSClipView (WebCoreView)
- (NSView *)_webcore_effectiveFirstResponder;
@end

@interface NSScrollView (WebCoreView)
- (NSView *)_webcore_effectiveFirstResponder;
@end

@implementation NSView (WebCoreView)

- (NSView *)_webcore_effectiveFirstResponder
{
    return self;
}

@end

@implementation NSClipView (WebCoreView)

- (NSView *)_webcore_effectiveFirstResponder
{
    NSView *view = [self documentView];
    return view ? [view _webcore_effectiveFirstResponder] : [super _webcore_effectiveFirstResponder];
}

@end

@implementation NSScrollView (WebCoreView)

- (NSView *)_webcore_effectiveFirstResponder
{
    NSView *view = [self contentView];
    return view ? [view _webcore_effectiveFirstResponder] : [super _webcore_effectiveFirstResponder];
}

@end

#endif // USE(APPKIT)
