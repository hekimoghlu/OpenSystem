/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#if HAVE(SCREEN_CAPTURE_KIT)

#import <ScreenCaptureKit/ScreenCaptureKit.h>

#if USE(APPLE_INTERNAL_SDK)

#import <ScreenCaptureKit/SCContentFilterPrivate.h>
#import <ScreenCaptureKit/SCContentSharingSession.h>
#import <ScreenCaptureKit/SCStream_Private.h>

#if HAVE(SC_CONTENT_SHARING_PICKER)
#if __has_include(<ScreenCaptureKit/SCContentSharingPicker.h>)
#import <ScreenCaptureKit/SCContentSharingPicker.h>
#endif
#endif // HAVE(SC_CONTENT_SHARING_PICKER)

#else // USE(APPLE_INTERNAL_SDK)

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, SCContentFilterType) {
    SCContentFilterTypeNothing,
    SCContentFilterTypeDisplay,
    SCContentFilterTypeAppsAndWindowsPinnedToDisplay,
    SCContentFilterTypeDesktopIndependentWindow,
    SCContentFilterTypeClientShouldImplementDefault
};

@interface SCContentFilterDesktopIndependentWindowInformation : NSObject
@property (nonatomic, readonly) SCWindow *window;
@end

@interface SCContentFilterDisplayInformation : NSObject
@property (nonatomic, readonly) SCDisplay *display;
@end

@interface SCContentFilter ()
@property (nonatomic, strong) SCWindow *window;
@property (nonatomic, strong) SCDisplay *display;
@property (nonatomic, assign) SCContentFilterType type;
@property (nullable, nonatomic, readonly) SCContentFilterDesktopIndependentWindowInformation *desktopIndependentWindowInfo;
@property (nullable, nonatomic, readonly) SCContentFilterDisplayInformation *displayInfo;
@end

@class SCContentSharingSession;

@protocol SCContentSharingSessionProtocol <NSObject>
@optional
- (void)sessionDidEnd:(SCContentSharingSession *)session;
- (void)sessionDidChangeContent:(SCContentSharingSession *)session;
- (void)pickerCanceledForSession:(SCContentSharingSession *)session;
@end

@interface SCContentSharingSession : NSObject
@property (nonatomic, weak) id<SCContentSharingSessionProtocol> delegate;
@property (nonatomic, readonly, copy) SCContentFilter *content;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithTitle:(NSString *)title;
- (void)showPickerForType:(SCContentFilterType)type;
- (void)end;
@end

@interface SCStream (SCContentSharing) <SCContentSharingSessionProtocol>
- (instancetype)initWithSharingSession:(SCContentSharingSession *)session captureOutputProperties:(SCStreamConfiguration *)streamConfig delegate:(id<SCStreamDelegate>)delegate;
@end

NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)

#endif // HAVE(SCREEN_CAPTURE_KIT)
