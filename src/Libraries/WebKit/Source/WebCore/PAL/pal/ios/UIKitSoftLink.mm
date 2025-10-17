/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#import "config.h"

#if PLATFORM(IOS_FAMILY)

#import <pal/spi/ios/UIKitSPI.h>
#import <wtf/SoftLinking.h>

@class CUICatalog;

SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_FLAGS(PAL, UIKit, RTLD_FIRST | RTLD_NOW)

SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIAccessibilityAnnouncementNotification, UIAccessibilityNotifications)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIApplicationWillResignActiveNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIApplicationWillEnterForegroundNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIApplicationDidBecomeActiveNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIApplicationDidEnterBackgroundNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIContentSizeCategoryDidChangeNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIApplicationDidChangeStatusBarOrientationNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIFontTextStyleCallout, UIFontTextStyle)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UIPasteboardNameGeneral, UIPasteboardName)
SOFT_LINK_CONSTANT_FOR_SOURCE(PAL, UIKit, UITextEffectsBeneathStatusBarWindowLevel, UIWindowLevel)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, NSParagraphStyle)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, NSPresentationIntent)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, UIKit, NSShadow, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, NSTextList)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIApplication)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, UIKit, UIColor, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIDevice)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIDocumentInteractionController)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIFont)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIGraphicsImageRenderer)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, UIKit, UIImage, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIImageView)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIImageSymbolConfiguration)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIFocusRingStyle)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UILabel)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIPasteboard)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIScreen)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UITapGestureRecognizer)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UITextEffectsWindow)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UITraitCollection)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIView)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIViewController)
SOFT_LINK_CLASS_FOR_SOURCE(PAL, UIKit, UIWindow)
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, UIKit, _UIKitGetTextEffectsCatalog, CUICatalog *, (void), ())
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, UIKit, UIAccessibilityIsGrayscaleEnabled, BOOL, (void), ())
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, UIKit, UIAccessibilityIsInvertColorsEnabled, BOOL, (void), ())
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, UIKit, UIAccessibilityIsReduceMotionEnabled, BOOL, (void), (), PAL_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, UIKit, UIAccessibilityDarkerSystemColorsEnabled, BOOL, (void), (), PAL_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE_WITH_EXPORT(PAL, UIKit, UIAccessibilityIsOnOffSwitchLabelsEnabled, BOOL, (void), (), PAL_EXPORT)
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, UIKit, UIAccessibilityPostNotification, void, (UIAccessibilityNotifications n, id argument), (n, argument))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, UIKit, UIGraphicsGetCurrentContext, CGContextRef, (void), ())
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, UIKit, UIGraphicsPopContext, void, (void), ())
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, UIKit, UIGraphicsPushContext, void, (CGContextRef context), (context))
SOFT_LINK_FUNCTION_FOR_SOURCE(PAL, UIKit, UIImagePNGRepresentation, NSData *, (UIImage *image), (image))

#endif
