/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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
#import "WKSelectMenuListViewController.h"

#if HAVE(PEPPER_UI_CORE)

#import "PepperUICoreSPI.h"
#import <wtf/RetainPtr.h>

static const CGFloat checkmarkImageViewWidth = 32;
static const CGFloat selectMenuItemHorizontalMargin = 9;
static const CGFloat selectMenuItemCellHeight = 44;
static NSString * const selectMenuCellReuseIdentifier = @"WebKitSelectMenuItemCell";

typedef NS_ENUM(NSInteger, PUICQuickboardListSection) {
    PUICQuickboardListSectionHeaderView,
    PUICQuickboardListSectionTrayButtons,
    PUICQuickboardListSectionTextOptions,
    PUICQuickboardListSectionContentUnavailable,
};

static constexpr CGFloat itemCellTopToLabelBaseline = 26;
static constexpr CGFloat itemCellBaselineToBottom = 8;

// FIXME: This can be removed when <rdar://problem/57807445> lands in a build.
@interface WKSelectMenuItemCell : PUICQuickboardListItemCell
@property (nonatomic, readonly) UIImageView *imageView;
@end

@implementation WKSelectMenuItemCell {
    RetainPtr<UIImageView> _imageView;
}

- (instancetype)initWithStyle:(UITableViewCellStyle)style reuseIdentifier:(NSString *)reuseIdentifier
{
    if (!(self = [super initWithStyle:style reuseIdentifier:reuseIdentifier]))
        return nil;

    _imageView = adoptNS([[UIImageView alloc] init]);
    UIImage *checkmarkImage = [PUICResources imageNamed:@"UIPreferencesBlueCheck" inBundle:[NSBundle bundleWithIdentifier:@"com.apple.PepperUICore"] shouldCache:YES];
    [_imageView setImage:[checkmarkImage _flatImageWithColor:[UIColor systemBlueColor]]];
    [_imageView setContentMode:UIViewContentModeCenter];
    [_imageView setHidden:YES];
    [self.contentView addSubview:_imageView.get()];
    return self;
}

- (UIImageView *)imageView
{
    return _imageView.get();
}

- (CGFloat)topToLabelBaselineSpecValue
{
    return itemCellTopToLabelBaseline;
}

- (CGFloat)baselineToBottomSpecValue
{
    return itemCellBaselineToBottom;
}

@end

#if HAVE(QUICKBOARD_COLLECTION_VIEWS)

@interface WKSelectMenuCollectionViewItemCell : PUICQuickboardListCollectionViewItemCell
@property (nonatomic, readonly) UIImageView *imageView;
@end

@implementation WKSelectMenuCollectionViewItemCell {
    RetainPtr<UIImageView> _imageView;
}

- (instancetype)initWithFrame:(CGRect)frame
{
    if (!(self = [super initWithFrame:frame]))
        return nil;

    _imageView = adoptNS([[UIImageView alloc] init]);
    UIImage *checkmarkImage = [PUICResources imageNamed:@"UIPreferencesBlueCheck" inBundle:[NSBundle bundleWithIdentifier:@"com.apple.PepperUICore"] shouldCache:YES];
    [_imageView setImage:[checkmarkImage _flatImageWithColor:[UIColor systemBlueColor]]];
    [_imageView setContentMode:UIViewContentModeCenter];
    [_imageView setHidden:YES];
    [self.contentView addSubview:_imageView.get()];
    return self;
}

- (UIImageView *)imageView
{
    return _imageView.get();
}

- (CGFloat)topToLabelBaselineSpecValue
{
    return itemCellTopToLabelBaseline;
}

- (CGFloat)baselineToBottomSpecValue
{
    return itemCellBaselineToBottom;
}

@end

#endif // HAVE(QUICKBOARD_COLLECTION_VIEWS)

@implementation WKSelectMenuListViewController {
    BOOL _isMultipleSelect;
    RetainPtr<NSMutableIndexSet> _indicesOfCheckedOptions;
}

@dynamic delegate;

- (instancetype)initWithDelegate:(id <WKSelectMenuListViewControllerDelegate>)delegate
{
    self = [super initWithDelegate:delegate dictationMode:PUICDictationModeText];
    return self;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    self.view.backgroundColor = UIColor.systemBackgroundColor;

    self.cancelButton.hidden = YES;
    self.showsAcceptButton = YES;

    _isMultipleSelect = [self.delegate selectMenuUsesMultipleSelection:self];
    _indicesOfCheckedOptions = adoptNS([[NSMutableIndexSet alloc] init]);
    for (NSInteger index = 0; index < self.numberOfListItems; ++index) {
        if ([self.delegate selectMenu:self hasSelectedOptionAtIndex:index])
            [_indicesOfCheckedOptions addIndex:index];
    }
}

#pragma mark - Quickboard subclassing

- (void)acceptButtonTappedWithCompletion:(PUICQuickboardCompletionBlock)completion
{
    completion(nil);
}

- (BOOL)shouldShowTrayView
{
    return NO;
}

// FIXME: This method can be removed when <rdar://problem/57807445> lands in a build.
- (void)didSelectListItem:(NSInteger)itemNumber
{
    [self didSelectListItemAtIndexPath:[NSIndexPath indexPathForRow:itemNumber inSection:PUICQuickboardListSectionTextOptions]];
}

- (void)didSelectListItemAtIndexPath:(NSIndexPath *)indexPath
{
    NSMutableArray *indexPathsToReload = [NSMutableArray array];
    NSInteger itemNumber = indexPath.row;
    if (_isMultipleSelect) {
        BOOL addIndex = ![_indicesOfCheckedOptions containsIndex:itemNumber];
        if (addIndex)
            [_indicesOfCheckedOptions addIndex:itemNumber];
        else
            [_indicesOfCheckedOptions removeIndex:itemNumber];
        [self.delegate selectMenu:self didCheckItemAtIndex:itemNumber checked:addIndex];
        [indexPathsToReload addObject:[NSIndexPath indexPathForRow:itemNumber inSection:indexPath.section]];
    } else {
        NSInteger previousSelectedIndex = [_indicesOfCheckedOptions firstIndex];
        if (previousSelectedIndex != itemNumber) {
            [_indicesOfCheckedOptions removeAllIndexes];
            [_indicesOfCheckedOptions addIndex:itemNumber];
            [self.delegate selectMenu:self didSelectItemAtIndex:itemNumber];
            if (previousSelectedIndex != NSNotFound)
                [indexPathsToReload addObject:[NSIndexPath indexPathForRow:previousSelectedIndex inSection:indexPath.section]];
            [indexPathsToReload addObject:[NSIndexPath indexPathForRow:itemNumber inSection:indexPath.section]];
        }
    }

    if (!indexPathsToReload.count)
        return;

#if HAVE(QUICKBOARD_COLLECTION_VIEWS)
    [self.collectionView reloadItemsAtIndexPaths:indexPathsToReload];
#endif
}

- (NSInteger)numberOfListItems
{
    return [self.delegate numberOfItemsInSelectMenu:self];
}

- (CGFloat)heightForListItem:(NSInteger)itemNumber width:(CGFloat)width
{
    return selectMenuItemCellHeight;
}

- (NSString *)listItemCellReuseIdentifier
{
    return selectMenuCellReuseIdentifier;
}

- (BOOL)shouldShowLanguageButton
{
    return NO;
}

#if HAVE(QUICKBOARD_COLLECTION_VIEWS)

- (Class)listItemCellClass
{
    return [WKSelectMenuCollectionViewItemCell class];
}

- (PUICQuickboardListCollectionViewItemCell *)itemCellForListItem:(NSInteger)itemNumber forIndexPath:(NSIndexPath *)indexPath
{
    auto reusableCell = retainPtr([self.collectionView dequeueReusableCellWithReuseIdentifier:selectMenuCellReuseIdentifier forIndexPath:indexPath]);

    [reusableCell bodyLabel].numberOfLines = 1;
    [reusableCell bodyLabel].lineBreakMode = NSLineBreakByTruncatingTail;
    [reusableCell bodyLabel].allowsDefaultTighteningForTruncation = YES;
    [reusableCell imageView].frame = UIRectInset([reusableCell contentView].bounds, 0, 0, 0, CGRectGetWidth([reusableCell contentView].bounds) - checkmarkImageViewWidth);
    [reusableCell setText:[self.delegate selectMenu:self displayTextForItemAtIndex:itemNumber]];

    if ([_indicesOfCheckedOptions containsIndex:itemNumber]) {
        [reusableCell bodyLabel].frame = UIRectInset([reusableCell contentView].bounds, 0, selectMenuItemHorizontalMargin + checkmarkImageViewWidth, 0, selectMenuItemHorizontalMargin);
        [reusableCell imageView].hidden = NO;
    } else {
        [reusableCell bodyLabel].frame = UIRectInset([reusableCell contentView].bounds, 0, selectMenuItemHorizontalMargin, 0, selectMenuItemHorizontalMargin);
        [reusableCell imageView].hidden = YES;
    }

    return reusableCell.autorelease();
}

- (BOOL)collectionViewSectionIsRadioSection:(NSInteger)sectionNumber
{
    return !_isMultipleSelect;
}

#endif // HAVE(QUICKBOARD_COLLECTION_VIEWS)

- (void)selectItemAtIndex:(NSInteger)index
{
#if HAVE(QUICKBOARD_COLLECTION_VIEWS)
    NSInteger itemSection = 0;
#else
    NSInteger itemSection = PUICQuickboardListSectionTextOptions;
#endif
    [self didSelectListItemAtIndexPath:[NSIndexPath indexPathForRow:index inSection:itemSection]];
}

@end

#endif // HAVE(PEPPER_UI_CORE)
