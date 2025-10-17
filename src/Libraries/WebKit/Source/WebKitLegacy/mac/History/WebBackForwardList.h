/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
#import <Foundation/Foundation.h>
#import <WebKitLegacy/WebKitAvailability.h>

@class WebHistoryItem;
@class WebBackForwardListPrivate;

/*!
    @class WebBackForwardList
    WebBackForwardList holds an ordered list of WebHistoryItems that comprises the back and
    forward lists.
    
    Note that the methods which modify instances of this class do not cause
    navigation to happen in other layers of the stack;  they are only for maintaining this data
    structure.
*/
WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface WebBackForwardList : NSObject {
@package
    WebBackForwardListPrivate *_private;
}

/*!
    @method addItem:
    @abstract Adds an entry to the list.
    @param item The entry to add.
    @discussion The added entry is inserted immediately after the current entry.
    If the current position in the list is not at the end of the list, elements in the
    forward list will be dropped at this point.  In addition, entries may be dropped to keep
    the size of the list within the maximum size.
*/    
- (void)addItem:(WebHistoryItem *)item;

/*!
    @method goBack
    @abstract Move the current pointer back to the entry before the current entry.
*/
- (void)goBack;

/*!
    @method goForward
    @abstract Move the current pointer ahead to the entry after the current entry.
*/
- (void)goForward;

/*!
    @method goToItem:
    @abstract Move the current pointer to the given entry.
    @param item The history item to move the pointer to
*/
- (void)goToItem:(WebHistoryItem *)item;

/*!
    @property backItem
    @abstract The entry right before the current entry, or nil if there isn't one.
*/
@property (nonatomic, readonly, strong) WebHistoryItem *backItem;

/*!
    @property currentItem
    @abstract Returns the current entry.
*/
@property (nonatomic, readonly, strong) WebHistoryItem *currentItem;

/*!
    @property forwardItem
    @abstract The entry right after the current entry, or nil if there isn't one.
*/
@property (nonatomic, readonly, strong) WebHistoryItem *forwardItem;

/*!
    @method backListWithLimit:
    @abstract Returns a portion of the list before the current entry.
    @param limit A cap on the size of the array returned.
    @result An array of items before the current entry, or nil if there are none.  The entries are in the order that they were originally visited.
*/
- (NSArray *)backListWithLimit:(int)limit;

/*!
    @method forwardListWithLimit:
    @abstract Returns a portion of the list after the current entry.
    @param limit A cap on the size of the array returned.
    @result An array of items after the current entry, or nil if there are none.  The entries are in the order that they were originally visited.
*/
- (NSArray *)forwardListWithLimit:(int)limit;

/*!
    @property capacity
    @abstract The list's maximum size.
*/
@property (nonatomic) int capacity;

/*!
    @property backListCount
    @abstract The number of items in the list.
*/
@property (nonatomic, readonly) int backListCount;

/*!
    @property forwardListCount
    @result The number of items in the list.
*/
@property (nonatomic, readonly) int forwardListCount;

/*!
    @method containsItem:
    @param item The item that will be checked for presence in the WebBackForwardList.
    @result Returns YES if the item is in the list. 
*/
- (BOOL)containsItem:(WebHistoryItem *)item;

/*!
    @method itemAtIndex:
    @abstract Returns an entry the given distance from the current entry.
    @param index Index of the desired list item relative to the current item; 0 is current item, -1 is back item, 1 is forward item, etc.
    @result The entry the given distance from the current entry. If index exceeds the limits of the list, nil is returned.
*/
- (WebHistoryItem *)itemAtIndex:(int)index;

@end

@interface WebBackForwardList(WebBackForwardListDeprecated)

// The following methods are deprecated, and exist for backward compatibility only.
// Use -[WebPreferences setUsesPageCache] and -[WebPreferences usesPageCache]
// instead.

/*!
    @method setPageCacheSize:
    @abstract The size passed to this method determines whether the WebView 
    associated with this WebBackForwardList will use the shared page cache.
    @param size If size is 0, the WebView associated with this WebBackForwardList
    will not use the shared page cache. Otherwise, it will.
*/
- (void)setPageCacheSize:(NSUInteger)size;

/*!
    @method pageCacheSize
    @abstract Returns the size of the shared page cache, or 0.
    @result The size of the shared page cache (in pages), or 0 if the WebView 
    associated with this WebBackForwardList will not use the shared page cache.
*/
- (NSUInteger)pageCacheSize;
@end
