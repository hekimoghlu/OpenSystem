/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
#pragma once

#if HAVE(PEPPER_UI_CORE)

#import <UIKit/UIKit.h>

typedef NS_ENUM(NSInteger, WKNumberPadButtonMode) {
    WKNumberPadButtonModeDefault,
    WKNumberPadButtonModeAlternate
};

typedef NS_ENUM(NSInteger, WKNumberPadButtonPosition) {
    WKNumberPadButtonPositionBottomLeft = -2,
    WKNumberPadButtonPositionBottomRight = -1,
    WKNumberPadButtonPosition0 = 0,
    WKNumberPadButtonPosition1 = 1,
    WKNumberPadButtonPosition2 = 2,
    WKNumberPadButtonPosition3 = 3,
    WKNumberPadButtonPosition4 = 4,
    WKNumberPadButtonPosition5 = 5,
    WKNumberPadButtonPosition6 = 6,
    WKNumberPadButtonPosition7 = 7,
    WKNumberPadButtonPosition8 = 8,
    WKNumberPadButtonPosition9 = 9
};

typedef NS_ENUM(NSInteger, WKNumberPadKey) {
    WKNumberPadKeyDash = -9,
    WKNumberPadKeyAsterisk = -8,
    WKNumberPadKeyOctothorpe = -7,
    WKNumberPadKeyClosingParenthesis = -6,
    WKNumberPadKeyOpeningParenthesis = -5,
    WKNumberPadKeyPlus = -4,
    WKNumberPadKeyAccept = -3,
    WKNumberPadKeyToggleMode = -2,
    WKNumberPadKeyNone = -1,
    WKNumberPadKey0 = WKNumberPadButtonPosition0,
    WKNumberPadKey1 = WKNumberPadButtonPosition1,
    WKNumberPadKey2 = WKNumberPadButtonPosition2,
    WKNumberPadKey3 = WKNumberPadButtonPosition3,
    WKNumberPadKey4 = WKNumberPadButtonPosition4,
    WKNumberPadKey5 = WKNumberPadButtonPosition5,
    WKNumberPadKey6 = WKNumberPadButtonPosition6,
    WKNumberPadKey7 = WKNumberPadButtonPosition7,
    WKNumberPadKey8 = WKNumberPadButtonPosition8,
    WKNumberPadKey9 = WKNumberPadButtonPosition9
};

@class WKNumberPadViewController;

@interface WKNumberPadView : UIView
- (instancetype)initWithFrame:(CGRect)frame controller:(WKNumberPadViewController *)controller NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithFrame:(CGRect)frame NS_UNAVAILABLE;
- (instancetype)initWithCoder:(NSCoder *)decoder NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
@end

#endif // HAVE(PEPPER_UI_CORE)
