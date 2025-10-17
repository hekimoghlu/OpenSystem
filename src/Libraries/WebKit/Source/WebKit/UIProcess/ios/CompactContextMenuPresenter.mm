/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#import "CompactContextMenuPresenter.h"

#if USE(UICONTEXTMENU)

#import <wtf/TZoneMallocInlines.h>

@interface UIContextMenuInteraction (SPI)
- (void)_presentMenuAtLocation:(CGPoint)location;
@end

#if HAVE(UI_BUTTON_PERFORM_PRIMARY_ACTION)
@interface UIButton (Staging_112292156)
- (void)performPrimaryAction;
@end
#endif

@interface WKCompactContextMenuPresenterButton : UIButton
@property (nonatomic, weak) id<UIContextMenuInteractionDelegate> externalDelegate;
@end

@implementation WKCompactContextMenuPresenterButton

- (UIContextMenuConfiguration *)contextMenuInteraction:(UIContextMenuInteraction *)interaction configurationForMenuAtLocation:(CGPoint)location
{
    if ([_externalDelegate respondsToSelector:@selector(contextMenuInteraction:configurationForMenuAtLocation:)])
        return [_externalDelegate contextMenuInteraction:interaction configurationForMenuAtLocation:location];

    return [super contextMenuInteraction:interaction configurationForMenuAtLocation:location];
}

- (UITargetedPreview *)contextMenuInteraction:(UIContextMenuInteraction *)interaction configuration:(UIContextMenuConfiguration *)configuration highlightPreviewForItemWithIdentifier:(id<NSCopying>)identifier
{
    if ([_externalDelegate respondsToSelector:@selector(contextMenuInteraction:configuration:highlightPreviewForItemWithIdentifier:)])
        return [_externalDelegate contextMenuInteraction:interaction configuration:configuration highlightPreviewForItemWithIdentifier:identifier];

    return [super contextMenuInteraction:interaction configuration:configuration highlightPreviewForItemWithIdentifier:identifier];
}

- (void)contextMenuInteraction:(UIContextMenuInteraction *)interaction willDisplayMenuForConfiguration:(UIContextMenuConfiguration *)configuration animator:(id<UIContextMenuInteractionAnimating>)animator
{
    [super contextMenuInteraction:interaction willDisplayMenuForConfiguration:configuration animator:animator];

    if ([_externalDelegate respondsToSelector:@selector(contextMenuInteraction:willDisplayMenuForConfiguration:animator:)])
        [_externalDelegate contextMenuInteraction:interaction willDisplayMenuForConfiguration:configuration animator:animator];
}

- (void)contextMenuInteraction:(UIContextMenuInteraction *)interaction willEndForConfiguration:(UIContextMenuConfiguration *)configuration animator:(id<UIContextMenuInteractionAnimating>)animator
{
    [super contextMenuInteraction:interaction willEndForConfiguration:configuration animator:animator];

    if ([_externalDelegate respondsToSelector:@selector(contextMenuInteraction:willEndForConfiguration:animator:)])
        [_externalDelegate contextMenuInteraction:interaction willEndForConfiguration:configuration animator:animator];
}

@end

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CompactContextMenuPresenter);

CompactContextMenuPresenter::CompactContextMenuPresenter(UIView *rootView, id<UIContextMenuInteractionDelegate> delegate)
    : m_rootView(rootView)
    , m_button([WKCompactContextMenuPresenterButton buttonWithType:UIButtonTypeSystem])
{
    [m_button setExternalDelegate:delegate];
    [m_button layer].zPosition = CGFLOAT_MIN;
    [m_button setHidden:YES];
    [m_button setUserInteractionEnabled:NO];
    [m_button setContextMenuInteractionEnabled:YES];
    [m_button setShowsMenuAsPrimaryAction:YES];
}

CompactContextMenuPresenter::~CompactContextMenuPresenter()
{
    [UIView performWithoutAnimation:^{
        dismiss();
    }];
    [m_button removeFromSuperview];
}

void CompactContextMenuPresenter::present(CGPoint locationInRootView)
{
    present(CGRect { locationInRootView, CGSizeZero });
}

UIContextMenuInteraction *CompactContextMenuPresenter::interaction() const
{
    return [m_button contextMenuInteraction];
}

void CompactContextMenuPresenter::present(CGRect rectInRootView)
{
    if (!m_rootView.window)
        return;

    [m_button setFrame:rectInRootView];
    if (![m_button superview])
        [m_rootView addSubview:m_button.get()];

#if HAVE(UI_BUTTON_PERFORM_PRIMARY_ACTION)
    static BOOL canPerformPrimaryAction = [UIButton instancesRespondToSelector:@selector(performPrimaryAction)];
    if (canPerformPrimaryAction) {
        [m_button performPrimaryAction];
        return;
    }
#endif

    [interaction() _presentMenuAtLocation:CGPointMake(CGRectGetMidX(rectInRootView), CGRectGetMidY(rectInRootView))];
}

void CompactContextMenuPresenter::dismiss()
{
    [[m_button contextMenuInteraction] dismissMenu];
}

void CompactContextMenuPresenter::updateVisibleMenu(UIMenu *(^updateBlock)(UIMenu *))
{
    [interaction() updateVisibleMenuWithBlock:updateBlock];
}

} // namespace WebKit

#endif // USE(UICONTEXTMENU)
