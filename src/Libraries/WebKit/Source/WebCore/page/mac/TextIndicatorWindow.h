/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#pragma once

#import "TextIndicator.h"
#import <wtf/CheckedPtr.h>
#import <wtf/Noncopyable.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMalloc.h>

OBJC_CLASS NSView;
OBJC_CLASS WebTextIndicatorLayer;

namespace WebCore {

#if PLATFORM(MAC)

class TextIndicatorWindow final : public CanMakeCheckedPtr<TextIndicatorWindow> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(TextIndicatorWindow, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(TextIndicatorWindow);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextIndicatorWindow);
public:
    WEBCORE_EXPORT explicit TextIndicatorWindow(NSView *);
    WEBCORE_EXPORT ~TextIndicatorWindow();

    WEBCORE_EXPORT void setTextIndicator(Ref<TextIndicator>, CGRect contentRect, TextIndicatorLifetime);
    WEBCORE_EXPORT void clearTextIndicator(TextIndicatorDismissalAnimation);

    WEBCORE_EXPORT void setAnimationProgress(float);

private:
    void closeWindow();

    void startFadeOut();

    NSView *m_targetView;
    RefPtr<TextIndicator> m_textIndicator;
    RetainPtr<NSWindow> m_textIndicatorWindow;
    RetainPtr<NSView> m_textIndicatorView;
    RetainPtr<WebTextIndicatorLayer> m_textIndicatorLayer;

    RunLoop::Timer m_temporaryTextIndicatorTimer;
};

#endif // PLATFORM(MAC)

} // namespace WebCore


