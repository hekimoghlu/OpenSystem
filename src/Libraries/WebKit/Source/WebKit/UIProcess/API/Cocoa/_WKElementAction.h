/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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

#import <WebKit/_WKActivatedElementInfo.h>

@class UIAction;
@class UIImage;

typedef NSString *UIActionIdentifier;
WK_EXTERN UIActionIdentifier const WKElementActionTypeToggleShowLinkPreviewsIdentifier;

typedef void (^WKElementActionHandler)(_WKActivatedElementInfo *);
typedef BOOL (^WKElementActionDismissalHandler)(void);

typedef NS_ENUM(NSInteger, _WKElementActionType) {
    _WKElementActionTypeCustom,
    _WKElementActionTypeOpen,
    _WKElementActionTypeCopy,
    _WKElementActionTypeSaveImage,
#if TARGET_OS_IOS || (defined(TARGET_OS_VISION) && TARGET_OS_VISION)
    _WKElementActionTypeAddToReadingList,
    _WKElementActionTypeOpenInDefaultBrowser WK_API_AVAILABLE(ios(9.0)),
    _WKElementActionTypeOpenInExternalApplication WK_API_AVAILABLE(ios(9.0)),
#endif
    _WKElementActionTypeShare WK_API_AVAILABLE(macos(10.12), ios(10.0)),
    _WKElementActionTypeOpenInNewTab WK_API_AVAILABLE(macos(10.15), ios(13.0)),
    _WKElementActionTypeOpenInNewWindow WK_API_AVAILABLE(macos(10.15), ios(13.0)),
    _WKElementActionTypeDownload WK_API_AVAILABLE(macos(10.15), ios(13.0)),
    _WKElementActionToggleShowLinkPreviews WK_API_AVAILABLE(macos(10.15), ios(13.0)),
    _WKElementActionTypeImageExtraction WK_API_AVAILABLE(ios(15.0)),
    _WKElementActionTypeRevealImage WK_API_AVAILABLE(ios(15.0)),
    _WKElementActionTypeCopyCroppedImage WK_API_AVAILABLE(ios(16.0)),
#if defined(TARGET_OS_VISION) && TARGET_OS_VISION && __VISION_OS_VERSION_MIN_REQUIRED >= 20000
    _WKElementActionTypeViewSpatial WK_API_AVAILABLE(visionos(2.2)),
#endif
    _WKElementActionPlayAnimation,
    _WKElementActionPauseAnimation,
} WK_API_AVAILABLE(macos(10.10), ios(8.0));

WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface _WKElementAction : NSObject

+ (instancetype)elementActionWithType:(_WKElementActionType)type;
+ (instancetype)elementActionWithType:(_WKElementActionType)type title:(NSString *)title actionHandler:(WKElementActionHandler)actionHandler WK_API_AVAILABLE(macos(10.15), ios(13.0));
+ (instancetype)elementActionWithType:(_WKElementActionType)type customTitle:(NSString *)title;
+ (instancetype)elementActionWithTitle:(NSString *)title actionHandler:(WKElementActionHandler)handler;

+ (UIImage *)imageForElementActionType:(_WKElementActionType)actionType WK_API_AVAILABLE(macos(10.15), ios(13.0));
+ (_WKElementActionType)elementActionTypeForUIActionIdentifier:(UIActionIdentifier)identifier WK_API_AVAILABLE(macos(10.15), ios(13.0));
- (UIAction *)uiActionForElementInfo:(_WKActivatedElementInfo *)elementInfo;

- (void)runActionWithElementInfo:(_WKActivatedElementInfo *)info WK_API_AVAILABLE(macos(10.15), ios(9.0));

@property (nonatomic, readonly) _WKElementActionType type;
@property (nonatomic, readonly) NSString* title;
@property (nonatomic, readonly) BOOL disabled;
@property (nonatomic, copy) WKElementActionDismissalHandler dismissalHandler;

@end

#endif // TARGET_OS_IPHONE
