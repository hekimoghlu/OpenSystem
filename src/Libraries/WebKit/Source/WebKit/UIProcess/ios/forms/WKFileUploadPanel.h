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
#if PLATFORM(IOS_FAMILY)

#import <UIKit/UIViewController.h>
#import <wtf/RetainPtr.h>

@class WKContentView;
@protocol WKFileUploadPanelDelegate;

namespace API {
class OpenPanelParameters;
}

namespace WebKit {
class WebOpenPanelResultListenerProxy;
enum class PickerDismissalReason : uint8_t;
enum class MovedSuccessfully : bool { No, Yes };
enum class KeyboardIsDismissing : bool { No, Yes };

struct TemporaryFileMoveResults {
    MovedSuccessfully operationResult;
    RetainPtr<NSURL> maybeMovedURL;
    RetainPtr<NSURL> temporaryDirectoryURL;
};
}

@interface WKFileUploadPanel : UIViewController
@property (nonatomic, weak) id <WKFileUploadPanelDelegate> delegate;
- (instancetype)initWithView:(WKContentView *)view;
- (void)presentWithParameters:(API::OpenPanelParameters*)parameters resultListener:(WebKit::WebOpenPanelResultListenerProxy*)listener;

- (BOOL)dismissIfNeededWithReason:(WebKit::PickerDismissalReason)reason;

#if USE(UICONTEXTMENU)
- (void)repositionContextMenuIfNeeded:(WebKit::KeyboardIsDismissing)isKeyboardBeingDismissed;
#endif

- (NSArray<NSString *> *)currentAvailableActionTitles;
- (NSArray<NSString *> *)acceptedTypeIdentifiers;

+ (WebKit::TemporaryFileMoveResults)_moveToNewTemporaryDirectory:(NSURL *)originalURL fileCoordinator:(NSFileCoordinator *)fileCoordinator fileManager:(NSFileManager *)fileManager asCopy:(BOOL)asCopy;
@end

@protocol WKFileUploadPanelDelegate <NSObject>
@optional
- (void)fileUploadPanelDidDismiss:(WKFileUploadPanel *)fileUploadPanel;
- (BOOL)fileUploadPanelDestinationIsManaged:(WKFileUploadPanel *)fileUploadPanel;
#if HAVE(PHOTOS_UI)
- (BOOL)fileUploadPanelPhotoPickerPrefersOriginalImageFormat:(WKFileUploadPanel *)fileUploadPanel;
#endif
@end

#endif // PLATFORM(IOS_FAMILY)
