/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
#ifndef WKImmediateActionController_h
#define WKImmediateActionController_h

#if PLATFORM(MAC)

#import "WKImmediateActionTypes.h"
#import "WebHitTestResultData.h"
#import <pal/spi/mac/NSImmediateActionGestureRecognizerSPI.h>
#import <wtf/CheckedPtr.h>
#import <wtf/NakedPtr.h>
#import <wtf/NakedRef.h>
#import <wtf/RetainPtr.h>

namespace WebKit {
class WebPageProxy;
class WebViewImpl;

enum class ImmediateActionState {
    None = 0,
    Pending,
    TimedOut,
    Ready
};
}

#if HAVE(SECURE_ACTION_CONTEXT)
@class DDSecureActionContext;
#else
@class DDActionContext;
#endif
@class QLPreviewMenuItem;

@interface WKImmediateActionController : NSObject <NSImmediateActionGestureRecognizerDelegate> {
@private
    WeakPtr<WebKit::WebPageProxy> _page;
    NSView *_view;
    WeakPtr<WebKit::WebViewImpl> _viewImpl;

    WebKit::ImmediateActionState _state;
    WebKit::WebHitTestResultData _hitTestResultData;
    BOOL _contentPreventsDefault;
    RefPtr<API::Object> _userData;
    uint32_t _type;
    RetainPtr<NSImmediateActionGestureRecognizer> _immediateActionRecognizer;

    BOOL _hasActivatedActionContext;
#if HAVE(SECURE_ACTION_CONTEXT)
    RetainPtr<DDSecureActionContext> _currentActionContext;
#else
    RetainPtr<DDActionContext> _currentActionContext;
#endif
    RetainPtr<QLPreviewMenuItem> _currentQLPreviewMenuItem;

    BOOL _hasActiveImmediateAction;
}

- (instancetype)initWithPage:(NakedRef<WebKit::WebPageProxy>)page view:(NSView *)view viewImpl:(NakedRef<WebKit::WebViewImpl>)viewImpl recognizer:(NSImmediateActionGestureRecognizer *)immediateActionRecognizer;
- (void)willDestroyView:(NSView *)view;
- (void)didPerformImmediateActionHitTest:(const WebKit::WebHitTestResultData&)hitTestResult contentPreventsDefault:(BOOL)contentPreventsDefault userData:(API::Object*)userData;
- (void)dismissContentRelativeChildWindows;
- (BOOL)hasActiveImmediateAction;

@end

#endif // PLATFORM(MAC)

#endif // WKImmediateActionController_h
