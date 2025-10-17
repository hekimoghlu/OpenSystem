/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#import "WKHighlightLongPressGestureRecognizer.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitUtilities.h"
#import <wtf/WeakObjCPtr.h>

@implementation WKHighlightLongPressGestureRecognizer {
    WeakObjCPtr<UIScrollView> _lastTouchedScrollView;
}

- (void)reset
{
    [super reset];

    _lastTouchedScrollView = nil;
}

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [super touchesBegan:touches withEvent:event];

    if (auto scrollView = WebKit::scrollViewForTouches(touches))
        _lastTouchedScrollView = scrollView;
}

- (UIScrollView *)lastTouchedScrollView
{
    return _lastTouchedScrollView.get().get();
}

@end

#endif // PLATFORM(IOS_FAMILY)
