/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#import "_WKWebViewTextInputNotifications.h"

#if HAVE(REDESIGNED_TEXT_CURSOR) && PLATFORM(MAC)

#import "WebPageProxy.h"
#import "WebViewImpl.h"
#import <pal/spi/mac/NSTextInputContextSPI.h>

@implementation _WKWebViewTextInputNotifications {
    WeakPtr<WebKit::WebViewImpl> _webView;
    WebCore::CaretAnimatorType _caretType;
}

- (WebCore::CaretAnimatorType)caretType
{
    return _caretType;
}

- (instancetype)initWithWebView:(WebKit::WebViewImpl*)webView
{
    if (!(self = [super init]))
        return nil;

    _webView = webView;
    _caretType = WebCore::CaretAnimatorType::Default;

    return self;
}

- (void)dictationDidStart
{
    if (_caretType == WebCore::CaretAnimatorType::Dictation) {
        if (_webView)
            _webView->protectedPage()->setCaretBlinkingSuspended(false);
        return;
    }

    _caretType = WebCore::CaretAnimatorType::Dictation;
    if (_webView) {
        if (NSTextInputContext *context = _webView->inputContext())
            context.showsCursorAccessories = YES;

        _webView->protectedPage()->setCaretAnimatorType(WebCore::CaretAnimatorType::Dictation);
    }
}

- (void)dictationDidEnd
{
    if (_caretType == WebCore::CaretAnimatorType::Default)
        return;

    _caretType = WebCore::CaretAnimatorType::Default;
    if (_webView)
        _webView->protectedPage()->setCaretAnimatorType(WebCore::CaretAnimatorType::Default);
}

- (void)dictationDidPause
{
    if (_webView)
        _webView->protectedPage()->setCaretBlinkingSuspended(true);
}

- (void)dictationDidResume
{
    if (_caretType == WebCore::CaretAnimatorType::Dictation) {
        if (_webView)
            _webView->protectedPage()->setCaretBlinkingSuspended(false);
        return;
    }

    _caretType = WebCore::CaretAnimatorType::Dictation;
    if (_webView)
        _webView->protectedPage()->setCaretAnimatorType(WebCore::CaretAnimatorType::Dictation);
}

@end

#endif
