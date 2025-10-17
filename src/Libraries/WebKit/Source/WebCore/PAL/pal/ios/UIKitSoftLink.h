/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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

#if PLATFORM(IOS_FAMILY)

#import <pal/spi/ios/UIKitSPI.h>
#import <wtf/SoftLinking.h>

@class CUICatalog;

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, UIKit)

SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIAccessibilityAnnouncementNotification, UIAccessibilityNotifications)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIApplicationWillResignActiveNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIApplicationWillEnterForegroundNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIApplicationDidBecomeActiveNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIApplicationDidEnterBackgroundNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIContentSizeCategoryDidChangeNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIApplicationDidChangeStatusBarOrientationNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIFontTextStyleCallout, UIFontTextStyle)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UIPasteboardNameGeneral, UIPasteboardName)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, UIKit, UITextEffectsBeneathStatusBarWindowLevel, UIWindowLevel)
SOFT_LINK_CLASS_FOR_HEADER(PAL, NSParagraphStyle)
SOFT_LINK_CLASS_FOR_HEADER(PAL, NSPresentationIntent)
SOFT_LINK_CLASS_FOR_HEADER(PAL, NSShadow)
SOFT_LINK_CLASS_FOR_HEADER(PAL, NSTextList)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIApplication)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIColor)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIDevice)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIDocumentInteractionController)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIFocusRingStyle)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIFont)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIGraphicsImageRenderer)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIImage)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIImageSymbolConfiguration)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIImageView)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UILabel)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIPasteboard)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIScreen)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UITapGestureRecognizer)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UITextEffectsWindow)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UITraitCollection)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIView)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIViewController)
SOFT_LINK_CLASS_FOR_HEADER(PAL, UIWindow)
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, _UIKitGetTextEffectsCatalog, CUICatalog *, (void), ())
#define _UIKitGetTextEffectsCatalog PAL::softLink_UIKit__UIKitGetTextEffectsCatalog
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIAccessibilityIsGrayscaleEnabled, BOOL, (void), ())
#define UIAccessibilityIsGrayscaleEnabled PAL::softLink_UIKit_UIAccessibilityIsGrayscaleEnabled
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIAccessibilityIsInvertColorsEnabled, BOOL, (void), ())
#define UIAccessibilityIsInvertColorsEnabled PAL::softLink_UIKit_UIAccessibilityIsInvertColorsEnabled
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIAccessibilityIsReduceMotionEnabled, BOOL, (void), ())
#define UIAccessibilityIsReduceMotionEnabled PAL::softLink_UIKit_UIAccessibilityIsReduceMotionEnabled
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIAccessibilityDarkerSystemColorsEnabled, BOOL, (void), ())
#define UIAccessibilityDarkerSystemColorsEnabled PAL::softLink_UIKit_UIAccessibilityDarkerSystemColorsEnabled
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIAccessibilityIsOnOffSwitchLabelsEnabled, BOOL, (void), ())
#define UIAccessibilityIsOnOffSwitchLabelsEnabled PAL::softLink_UIKit_UIAccessibilityIsOnOffSwitchLabelsEnabled
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIAccessibilityPostNotification, void, (UIAccessibilityNotifications n, id argument), (n, argument))
#define UIAccessibilityPostNotification PAL::softLink_UIKit_UIAccessibilityPostNotification
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIGraphicsGetCurrentContext, CGContextRef, (void), ())
#define UIGraphicsGetCurrentContext PAL::softLink_UIKit_UIGraphicsGetCurrentContext
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIGraphicsPopContext, void, (void), ())
#define UIGraphicsPopContext PAL::softLink_UIKit_UIGraphicsPopContext
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIGraphicsPushContext, void, (CGContextRef context), (context))
#define UIGraphicsPushContext PAL::softLink_UIKit_UIGraphicsPushContext
SOFT_LINK_FUNCTION_FOR_HEADER(PAL, UIKit, UIImagePNGRepresentation, NSData *, (UIImage *image), (image))
#define UIImagePNGRepresentation PAL::softLink_UIKit_UIImagePNGRepresentation

#endif
