/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
#import "WKFormSelectControl.h"

#if PLATFORM(IOS_FAMILY)

#import "WKContentView.h"
#import "WKContentViewInteraction.h"
#import "WKFormPopover.h"
#import "WKFormSelectPicker.h"
#import "WKFormSelectPopover.h"
#import "WebPageProxy.h"
#import <UIKit/UIPickerView.h>
#import <pal/system/ios/UserInterfaceIdiom.h>
#import <wtf/RetainPtr.h>

using namespace WebKit;

static const CGFloat minimumOptionFontSize = 12;

CGFloat adjustedFontSize(CGFloat textWidth, UIFont *font, CGFloat initialFontSize, const Vector<OptionItem>& items)
{
    CGFloat adjustedSize = initialFontSize;
    for (size_t i = 0; i < items.size(); ++i) {
        const OptionItem& item = items[i];
        if (item.text.isEmpty())
            continue;

        CGFloat actualFontSize = initialFontSize;
        [(NSString *)item.text _legacy_sizeWithFont:font minFontSize:minimumOptionFontSize actualFontSize:&actualFontSize forWidth:textWidth lineBreakMode:NSLineBreakByWordWrapping];

        if (actualFontSize > 0 && actualFontSize < adjustedSize)
            adjustedSize = actualFontSize;
    }
    return adjustedSize;
}

@implementation WKFormSelectControl {
    RetainPtr<NSObject<WKFormControl>> _control;
}

- (instancetype)initWithView:(WKContentView *)view
{
    bool hasGroups = false;
    for (size_t i = 0; i < view.focusedElementInformation.selectOptions.size(); ++i) {
        if (view.focusedElementInformation.selectOptions[i].isGroup) {
            hasGroups = true;
            break;
        }
    }

    RetainPtr<NSObject <WKFormControl>> control;

    if (view._shouldUseContextMenusForFormControls) {
        if (view.focusedElementInformation.isMultiSelect)
            control = adoptNS([[WKSelectMultiplePicker alloc] initWithView:view]);
        else
            control = adoptNS([[WKSelectPicker alloc] initWithView:view]);

        self = [super initWithView:view control:WTFMove(control)];
        return self;
    }

    if (!PAL::currentUserInterfaceIdiomIsSmallScreen())
        control = adoptNS([[WKSelectPopover alloc] initWithView:view hasGroups:hasGroups]);
    else if (view.focusedElementInformation.isMultiSelect || hasGroups)
        control = adoptNS([[WKMultipleSelectPicker alloc] initWithView:view]);
    else
        control = adoptNS([[WKSelectSinglePicker alloc] initWithView:view]);

    self = [super initWithView:view control:WTFMove(control)];
    return self;
}

@end

@implementation WKFormSelectControl(WKTesting)

- (void)selectRow:(NSInteger)rowIndex inComponent:(NSInteger)componentIndex extendingSelection:(BOOL)extendingSelection
{
    if ([self.control respondsToSelector:@selector(selectRow:inComponent:extendingSelection:)])
        [id<WKSelectTesting>(self.control) selectRow:rowIndex inComponent:componentIndex extendingSelection:extendingSelection];
}

- (NSString *)selectFormPopoverTitle
{
    if (auto *popover = dynamic_objc_cast<WKSelectPopover>(self.control))
        return [popover tableViewController].title;
    return nil;
}

- (BOOL)selectFormAccessoryHasCheckedItemAtRow:(long)rowIndex
{
    return [self.control respondsToSelector:@selector(selectFormAccessoryHasCheckedItemAtRow:)]
        && [id<WKSelectTesting>(self.control) selectFormAccessoryHasCheckedItemAtRow:rowIndex];
}

- (NSArray<NSString *> *)menuItemTitles
{
    if ([self.control respondsToSelector:@selector(menuItemTitles)])
        return [id<WKSelectTesting>(self.control) menuItemTitles];
    return nil;
}

@end

#endif  // PLATFORM(IOS_FAMILY)
