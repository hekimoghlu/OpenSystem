/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
#if HAVE(PEPPER_UI_CORE)

#import "WKQuickboardViewControllerDelegate.h"

@class WKSelectMenuListViewController;

@protocol WKSelectMenuListViewControllerDelegate <WKQuickboardViewControllerDelegate>

- (void)selectMenu:(WKSelectMenuListViewController *)selectMenu didSelectItemAtIndex:(NSUInteger)index;
- (void)selectMenu:(WKSelectMenuListViewController *)selectMenu didCheckItemAtIndex:(NSUInteger)index checked:(BOOL)checked;

- (BOOL)selectMenu:(WKSelectMenuListViewController *)selectMenu hasSelectedOptionAtIndex:(NSUInteger)index;
- (NSUInteger)numberOfItemsInSelectMenu:(WKSelectMenuListViewController *)selectMenu;
- (NSString *)selectMenu:(WKSelectMenuListViewController *)selectMenu displayTextForItemAtIndex:(NSUInteger)index;
- (BOOL)selectMenuUsesMultipleSelection:(WKSelectMenuListViewController *)selectMenu;

@end

@interface WKSelectMenuListViewController : PUICQuickboardListViewController

- (instancetype)initWithDelegate:(id <WKSelectMenuListViewControllerDelegate>)delegate NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithDelegate:(id <PUICQuickboardViewControllerDelegate>)delegate dictationMode:(PUICDictationMode)dictationMode NS_UNAVAILABLE;
- (instancetype)initWithCoder:(NSCoder *)coder NS_UNAVAILABLE;

@property (nonatomic, weak) id <WKSelectMenuListViewControllerDelegate> delegate;

@end

@interface WKSelectMenuListViewController (Testing)

- (void)selectItemAtIndex:(NSInteger)index;

@end

#endif // HAVE(PEPPER_UI_CORE)
