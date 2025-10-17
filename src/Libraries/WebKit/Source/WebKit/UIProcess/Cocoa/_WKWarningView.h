/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#import "WKFoundation.h"
#import <WebCore/ColorCocoa.h>
#import <variant>
#import <wtf/CompletionHandler.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>

#if PLATFORM(MAC)
#import <AppKit/AppKit.h>
#else
#import <UIKit/UIKit.h>
#endif

namespace WebKit {
class BrowsingWarning;
enum class ContinueUnsafeLoad : bool;
}

OBJC_CLASS _WKWarningViewTextView;

#if PLATFORM(MAC)
using ViewType = NSView;
using RectType = NSRect;
#else
using ViewType = UIView;
using RectType = CGRect;
#endif

@interface _WKWarningViewBox : ViewType {
@package
#if PLATFORM(MAC)
    RetainPtr<WebCore::CocoaColor> _backgroundColor;
#endif
}
- (void)setWarningViewBackgroundColor:(WebCore::CocoaColor *)color;
@end

#if PLATFORM(MAC)
@interface _WKWarningView : _WKWarningViewBox<NSTextViewDelegate, NSAccessibilityGroup>
#else
@interface _WKWarningView : UIScrollView<UITextViewDelegate>
#endif
{
@package
    CompletionHandler<void(std::variant<WebKit::ContinueUnsafeLoad, URL>&&)> _completionHandler;
    WeakObjCPtr<_WKWarningViewTextView> _details;
    WeakObjCPtr<_WKWarningViewBox> _box;
#if PLATFORM(WATCHOS)
    WeakObjCPtr<UIResponder> _previousFirstResponder;
#endif
}

@property (nonatomic, readonly) RefPtr<const WebKit::BrowsingWarning> warning;

- (instancetype)initWithFrame:(RectType)frame browsingWarning:(const WebKit::BrowsingWarning&)warning completionHandler:(CompletionHandler<void(std::variant<WebKit::ContinueUnsafeLoad, URL>&&)>&&)completionHandler;

- (BOOL)forMainFrameNavigation;

@end
