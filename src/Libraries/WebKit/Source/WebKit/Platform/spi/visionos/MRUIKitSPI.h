/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#if PLATFORM(VISION)

#import "RealitySimulationServicesSPI.h"

#if USE(APPLE_INTERNAL_SDK)

#import <MRUIKit/MRUIStage.h>
#import <MRUIKit/UIApplication+MRUIKit.h>
#import <MRUIKit/UIWindowScene+MRUIKit_Private.h>

#else

typedef NS_ENUM(NSUInteger, MRUIDarknessPreference) {
    MRUIDarknessPreferenceUnspecified = 0,
    MRUIDarknessPreferenceDim,
    MRUIDarknessPreferenceDark,
    MRUIDarknessPreferenceVeryDark,
};

@interface MRUIStage : NSObject
@property (nonatomic, readwrite) MRUIDarknessPreference preferredDarkness;
@end

typedef NS_ENUM(NSUInteger, MRUISceneResizingBehavior) {
    MRUISceneResizingBehaviorUnspecified = 0,
    MRUISceneResizingBehaviorNone,
    MRUISceneResizingBehaviorUniform,
    MRUISceneResizingBehaviorFreeform,
};

@interface MRUIWindowScenePlacement : NSObject
@property (nonatomic, assign) MRUISceneResizingBehavior preferredResizingBehavior;
@property (nonatomic, assign) RSSSceneChromeOptions preferredChromeOptions;
@end

typedef NS_ENUM(NSInteger, MRUICloseWindowSceneReason) {
    MRUICloseReasonCloseButton,
    MRUICloseReasonCommandKey,
};

@protocol MRUIWindowSceneDelegate <UIWindowSceneDelegate>
@optional
- (BOOL)windowScene:(UIWindowScene *)windowScene shouldCloseForReason:(MRUICloseWindowSceneReason)reason;
@end

@interface UIApplication (MRUIKit)
@property (nonatomic, readonly) MRUIStage *mrui_activeStage;
@end

@class MRUIWindowSceneResizeRequestOptions;

typedef void (^MRUIWindowSceneResizeRequestCompletion)(CGSize grantedSize, NSError *error);

@interface UIWindowScene (MRUIKit)

- (void)mrui_requestResizeToSize:(CGSize)size options:(MRUIWindowSceneResizeRequestOptions *)options completion:(MRUIWindowSceneResizeRequestCompletion)completion;

@property (nonatomic, readonly) MRUIWindowScenePlacement *mrui_placement;

@end

extern NSNotificationName const _MRUIWindowSceneDidBeginRepositioningNotification;
extern NSNotificationName const _MRUIWindowSceneDidEndRepositioningNotification;

#endif // USE(APPLE_INTERNAL_SDK)

// FIXME: <rdar://111655142> Import ornaments SPI using framework headers.

@interface MRUIPlatterOrnament : NSObject
@property (nonatomic, assign, getter=_depthDisplacement, setter=_setDepthDisplacement:) CGFloat depthDisplacement;
@property (nonatomic, assign) CGPoint offset2D;
@property (nonatomic, assign) CGPoint sceneAnchorPoint;
@property (nonatomic, readwrite, strong) UIViewController *viewController;
@end

@interface MRUIPlatterOrnamentManager : NSObject
@property (nonatomic, readonly) NSArray<MRUIPlatterOrnament *> *ornaments;
@end

@interface UIWindowScene (MRUIPlatterOrnaments)
@property (nonatomic, readonly) MRUIPlatterOrnamentManager *_mrui_platterOrnamentManager;
@end

#endif // PLATFORM(VISION)
