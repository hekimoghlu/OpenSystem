/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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
#import "WKModelInteractionGestureRecognizer.h"

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)

#import "RemoteLayerTreeViews.h"
#import "WKModelView.h"
#import <UIKit/UIGestureRecognizerSubclass.h>
#import <pal/spi/ios/SystemPreviewSPI.h>

@implementation WKModelInteractionGestureRecognizer

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    ASSERT([event touchesForGestureRecognizer:self].count);

    if (![self.view isKindOfClass:[WKModelView class]]) {
        [self setState:UIGestureRecognizerStateFailed];
        return;
    }

    [((WKModelView *)self.view).preview touchesBegan:touches withEvent:event];
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    [self setState:UIGestureRecognizerStateChanged];

    [((WKModelView *)self.view).preview touchesMoved:touches withEvent:event];
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    auto finalTouchesEnded = [touches isEqualToSet:[event touchesForGestureRecognizer:self]];
    [self setState:finalTouchesEnded ? UIGestureRecognizerStateEnded : UIGestureRecognizerStateChanged];

    [((WKModelView *)self.view).preview touchesEnded:touches withEvent:event];
}

- (void)touchesCancelled:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event
{
    auto finalTouchesCancelled = [touches isEqualToSet:[event touchesForGestureRecognizer:self]];
    [self setState:finalTouchesCancelled ? UIGestureRecognizerStateCancelled : UIGestureRecognizerStateChanged];

    [((WKModelView *)self.view).preview touchesCancelled:touches withEvent:event];
}

@end

#endif
