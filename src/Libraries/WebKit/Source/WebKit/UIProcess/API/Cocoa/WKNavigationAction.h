/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif

NS_ASSUME_NONNULL_BEGIN

@class WKFrameInfo;

/*! @enum WKNavigationType
 @abstract The type of action triggering a navigation.
 @constant WKNavigationTypeLinkActivated    A link with an href attribute was activated.
 @constant WKNavigationTypeFormSubmitted    A form was submitted.
 @constant WKNavigationTypeBackForward      An item from the back-forward list was requested.
 @constant WKNavigationTypeReload           The webpage was reloaded.
 @constant WKNavigationTypeFormResubmitted  A form was resubmitted (for example by going back, going forward, or reloading).
 @constant WKNavigationTypeOther            Navigation is taking place for some other reason.
 */
typedef NS_ENUM(NSInteger, WKNavigationType) {
    WKNavigationTypeLinkActivated,
    WKNavigationTypeFormSubmitted,
    WKNavigationTypeBackForward,
    WKNavigationTypeReload,
    WKNavigationTypeFormResubmitted,
    WKNavigationTypeOther = -1,
} WK_API_AVAILABLE(macos(10.10), ios(8.0));

/*! 
A WKNavigationAction object contains information about an action that may cause a navigation, used for making policy decisions.
 */
WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKNavigationAction : NSObject

/*! @abstract The frame requesting the navigation.
 */
@property (nonatomic, readonly, copy) WKFrameInfo *sourceFrame;

/*! @abstract The target frame, or nil if this is a new window navigation.
 */
@property (nullable, nonatomic, readonly, copy) WKFrameInfo *targetFrame;

/*! @abstract The type of action that triggered the navigation.
 @discussion The value is one of the constants of the enumerated type WKNavigationType.
 */
@property (nonatomic, readonly) WKNavigationType navigationType;

/*! @abstract The navigation's request.
 */
@property (nonatomic, readonly, copy) NSURLRequest *request;

/*! @abstract A value indicating whether the web content used a download attribute to indicate that this should be downloaded.
*/
@property (nonatomic, readonly) BOOL shouldPerformDownload WK_API_AVAILABLE(macos(11.3), ios(14.5));

#if TARGET_OS_IPHONE

/*! @abstract The modifier keys that were in effect when the navigation was requested.
 */
@property (nonatomic, readonly) UIKeyModifierFlags modifierFlags WK_API_AVAILABLE(ios(WK_IOS_TBA), visionos(WK_XROS_TBA));

/*! @abstract The button mask of the index of the mouse button causing the navigation to be requested.
 */
@property (nonatomic, readonly) UIEventButtonMask buttonNumber WK_API_AVAILABLE(ios(WK_IOS_TBA), visionos(WK_XROS_TBA));

#else

/*! @abstract The modifier keys that were in effect when the navigation was requested.
 */
@property (nonatomic, readonly) NSEventModifierFlags modifierFlags;

/*! @abstract The number of the mouse button causing the navigation to be requested.
 */
@property (nonatomic, readonly) NSInteger buttonNumber;

#endif

@end

NS_ASSUME_NONNULL_END
