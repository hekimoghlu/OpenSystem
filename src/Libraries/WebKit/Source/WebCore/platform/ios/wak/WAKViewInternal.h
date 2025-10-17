/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#if TARGET_OS_IPHONE

#import "WAKView.h"
#import "WKView.h"

@interface WAKView ()
{
@package
    WKViewContext viewContext;
    WKViewRef viewRef;

    // This array is only used to keep WAKViews alive.
    // The actual subviews are maintained by the WKView.
    NSMutableSet *subviewReferences;

    BOOL _isHidden;
    BOOL _drawsOwnDescendants;
}

- (WKViewRef)_viewRef;
+ (WAKView *)_wrapperForViewRef:(WKViewRef)_viewRef;
- (id)_initWithViewRef:(WKViewRef)view;
- (BOOL)_handleResponderCall:(WKViewResponderCallbackType)type;
- (NSMutableSet *)_subviewReferences;
- (BOOL)_selfHandleEvent:(WebEvent *)event;

@end

static inline WAKView *WAKViewForWKViewRef(WKViewRef view)
{
    if (!view)
        return nil;
    WAKView *wrapper = (WAKView *)view->wrapper;
    if (wrapper)
        return wrapper;
    return [WAKView _wrapperForViewRef:view];
}

#endif
