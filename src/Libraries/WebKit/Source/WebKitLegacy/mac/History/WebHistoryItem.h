/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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

#if !TARGET_OS_IPHONE
#import <AppKit/AppKit.h>
#endif

@class WebHistoryItemPrivate;
@class NSURL;

/*
    @discussion Notification sent when history item is modified.
    @constant WebHistoryItemChanged Posted from whenever the value of
    either the item's title, alternate title, url strings, or last visited interval
    changes.  The userInfo will be nil.
*/
extern NSString *WebHistoryItemChangedNotification WEBKIT_DEPRECATED_MAC(10_3, 10_14);

/*!
    @class WebHistoryItem
    @discussion  WebHistoryItems are created by WebKit to represent pages visited.
    The WebBackForwardList and WebHistory classes both use WebHistoryItems to represent
    pages visited.  With the exception of the displayTitle, the properties of 
    WebHistoryItems are set by WebKit.  WebHistoryItems are normally never created directly.
*/
WEBKIT_CLASS_DEPRECATED_MAC(10_3, 10_14)
@interface WebHistoryItem : NSObject <NSCopying>
{
@package
    WebHistoryItemPrivate *_private;
}

/*!
    @method initWithURLString:title:lastVisitedTimeInterval:
    @param URLString The URL string for the item.
    @param title The title to use for the item.  This is normally the <title> of a page.
    @param time The time used to indicate when the item was used.
    @abstract Initialize a new WebHistoryItem
    @discussion WebHistoryItems are normally created for you by the WebKit.
    You may use this method to prepopulate a WebBackForwardList, or create
    'artificial' items to add to a WebBackForwardList.  When first initialized
    the URLString and originalURLString will be the same.
*/
- (instancetype)initWithURLString:(NSString *)URLString title:(NSString *)title lastVisitedTimeInterval:(NSTimeInterval)time;

/*!
    @property originalURLString
    @abstract The string representation of the initial URL of this item.
    This value is normally set by the WebKit.
*/
@property (nonatomic, readonly, copy) NSString *originalURLString;

/*!
    @property URLString
    @abstract The string representation of the URL represented by this item.
    @discussion The URLString may be different than the originalURLString if the page
    redirected to a new location.  This value is normally set by the WebKit.
*/
@property (nonatomic, readonly, copy) NSString *URLString;


/*!
    @property title
    @abstract The title of the page represented by this item.
    @discussion This title cannot be changed by the client.  This value
    is normally set by the WebKit when a page title for the item is received.
*/
@property (nonatomic, readonly, copy) NSString *title;

/*!
    @property lastVisitedTimeInterval
    @abstract The last time the page represented by this item was visited. The interval
    is since the reference date as determined by NSDate.  This value is normally set by
    the WebKit.
*/
@property (nonatomic, readonly) NSTimeInterval lastVisitedTimeInterval;

/*
    @property alternateTitle
    @abstract A title that may be used by the client to display this item.
*/
@property (nonatomic, copy) NSString *alternateTitle;

#if !TARGET_OS_IPHONE
/*!
    @property icon
    @abstract The favorite icon of the page represented by this item.
    @discussion This icon returned will be determined by the WebKit.
*/
@property (nonatomic, readonly, strong) NSImage *icon;
#endif

@end
