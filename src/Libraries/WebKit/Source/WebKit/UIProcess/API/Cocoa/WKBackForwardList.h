/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#import <WebKit/WKFoundation.h>

#import <WebKit/WKBackForwardListItem.h>

/*! @abstract A WKBackForwardList object is a list of webpages previously
 visited in a web view that can be reached by going back or forward.
 */
NS_ASSUME_NONNULL_BEGIN

WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKBackForwardList : NSObject

/*! @abstract The current item.
 */
@property (nullable, nonatomic, readonly, strong) WKBackForwardListItem *currentItem;

/*! @abstract The item immediately preceding the current item, or nil
if there isn't one.
 */
@property (nullable, nonatomic, readonly, strong) WKBackForwardListItem *backItem;

/*! @abstract The item immediately following the current item, or nil
if there isn't one.
 */
@property (nullable, nonatomic, readonly, strong) WKBackForwardListItem *forwardItem;

/*! @abstract Returns the item at a specified distance from the current
 item.
 @param index Index of the desired list item relative to the current item:
 0 for the current item, -1 for the immediately preceding item, 1 for the
 immediately following item, and so on.
 @result The item at the specified distance from the current item, or nil
 if the index parameter exceeds the limits of the list.
 */
- (nullable WKBackForwardListItem *)itemAtIndex:(NSInteger)index;

/*! @abstract The portion of the list preceding the current item.
 @discussion The items are in the order in which they were originally
 visited.
 */
@property (nonatomic, readonly, copy) NSArray<WKBackForwardListItem *> *backList;

/*! @abstract The portion of the list following the current item.
 @discussion The items are in the order in which they were originally
 visited.
 */
@property (nonatomic, readonly, copy) NSArray<WKBackForwardListItem *> *forwardList;

@end

NS_ASSUME_NONNULL_END
