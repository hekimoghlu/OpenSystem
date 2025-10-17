/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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

#if USE(UICONTEXTMENU)

#import <UIKit/UIKit.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>

@class WKCompactContextMenuPresenterButton;

namespace WebKit {

class CompactContextMenuPresenter {
    WTF_MAKE_TZONE_ALLOCATED(CompactContextMenuPresenter);
    WTF_MAKE_NONCOPYABLE(CompactContextMenuPresenter);
public:
    CompactContextMenuPresenter(UIView *rootView, id<UIContextMenuInteractionDelegate>);
    ~CompactContextMenuPresenter();

    void present(CGRect rectInRootView);
    void present(CGPoint locationInRootView);
    void dismiss();

    void updateVisibleMenu(UIMenu *(^)(UIMenu *));

    UIContextMenuInteraction *interaction() const;

private:
    __weak UIView *m_rootView { nil };
    RetainPtr<WKCompactContextMenuPresenterButton> m_button;
};

} // namespace WebKit

#endif // USE(UICONTEXTMENU)
