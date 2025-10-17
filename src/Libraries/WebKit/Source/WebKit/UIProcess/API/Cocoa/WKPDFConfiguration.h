/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#import <CoreGraphics/CGGeometry.h>
#import <Foundation/Foundation.h>
#import <WebKit/WKFoundation.h>

NS_ASSUME_NONNULL_BEGIN

WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.15.4), ios(13.4))
@interface WKPDFConfiguration : NSObject <NSCopying>

/*! @abstract The rect to capture in web page coordinates
 @discussion If the rect is set to the null rect, the bounds of the currently displayed web page will be used.
 The initial value is the null rect.
 */
@property (nonatomic) CGRect rect NS_REFINED_FOR_SWIFT;

/*! @abstract A Boolean value indicating whether the PDF should allow transparent backgrounds.
 @discussion The default value is `NO`.
 */
@property (nonatomic) BOOL allowTransparentBackground WK_API_AVAILABLE(macos(14.0), ios(17.0));

@end

NS_ASSUME_NONNULL_END
