/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#import <WebKitLegacy/WebHistory.h>

/*
    @constant WebHistoryItemsDiscardedWhileLoadingNotification Posted from loadFromURL:error:.  
    This notification comes with a userInfo dictionary that contains the array of
    items discarded due to the date limit or item limit. The key for the array is WebHistoryItemsKey.
*/
// FIXME: This notification should become public API.
extern NSString *WebHistoryItemsDiscardedWhileLoadingNotification;

@interface WebHistory (WebPrivate)

// FIXME: The following SPI is used by Safari. Should it be made into public API?
- (WebHistoryItem *)_itemForURLString:(NSString *)URLString;

/*!
    @method allItems
    @result Returns an array of all WebHistoryItems in WebHistory, in an undefined order.
*/
- (NSArray *)allItems;

/*!
    @method _data
    @result A data object with the entire history in the same format used by the saveToURL:error: method.
*/
- (NSData *)_data;

+ (void)_setVisitedLinkTrackingEnabled:(BOOL)visitedLinkTrackingEnabled;
+ (void)_removeAllVisitedLinks;
@end
