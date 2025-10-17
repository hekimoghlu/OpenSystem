/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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

#import <Foundation/Foundation.h>

/*! WKWindowFeatures specifies optional attributes for the containing window when a new WKWebView is requested.
 */
NS_ASSUME_NONNULL_BEGIN

WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKWindowFeatures : NSObject

/*! @abstract BOOL. Whether the menu bar should be visible. nil if menu bar visibility was not specified.
 */
@property (nullable, nonatomic, readonly) NSNumber *menuBarVisibility;

/*! @abstract BOOL. Whether the status bar should be visible. nil if status bar visibility was not specified.
 */
@property (nullable, nonatomic, readonly) NSNumber *statusBarVisibility;

/*! @abstract BOOL. Whether toolbars should be visible. nil if toolbar visibility was not specified.
 */
@property (nullable, nonatomic, readonly) NSNumber *toolbarsVisibility;

/*! @abstract BOOL. Whether the containing window should be resizable. nil if resizability was not specified.
 */
@property (nullable, nonatomic, readonly) NSNumber *allowsResizing;

/*! @abstract CGFloat. The x coordinate of the containing window. nil if the x coordinate was not specified.
 */
@property (nullable, nonatomic, readonly) NSNumber *x;

/*! @abstract CGFloat. The y coordinate of the containing window. nil if the y coordinate was not specified.
 */
@property (nullable, nonatomic, readonly) NSNumber *y;

/*! @abstract CGFloat. The width coordinate of the containing window. nil if the width was not specified.
 */
@property (nullable, nonatomic, readonly) NSNumber *width;

/*! @abstract CGFloat. The height coordinate of the containing window. nil if the height was not specified.
 */
@property (nullable, nonatomic, readonly) NSNumber *height;

@end

NS_ASSUME_NONNULL_END
