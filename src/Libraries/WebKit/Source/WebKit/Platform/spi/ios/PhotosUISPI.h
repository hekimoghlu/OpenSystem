/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#if HAVE(PHOTOS_UI)

#import <PhotosUI/PhotosUI.h>

#if USE(APPLE_INTERNAL_SDK)

#if HAVE(PX_ACTIVITY_PROGRESS_CONTROLLER)
#import <PhotosUICore/PXActivityProgressController.h>
#else
#import <PhotosUIPrivate/PUActivityProgressController.h>
#endif

#import <PhotosUI/PHPicker_Private.h>

#else

#import "UIKitSPI.h"

#if HAVE(PX_ACTIVITY_PROGRESS_CONTROLLER)
@interface PXActivityProgressController : NSObject
#else
@interface PUActivityProgressController : NSObject
#endif

@property (nonatomic, copy) NSString *title;

@property (nonatomic, copy) void (^cancellationHandler)(void);

@property (nonatomic, readonly) BOOL isCancelled;

- (void)setFractionCompleted:(double)fractionCompleted;

- (void)showAnimated:(BOOL)animated allowDelay:(BOOL)allowDelay;
- (void)hideAnimated:(BOOL)animated allowDelay:(BOOL)allowDelay;

@end

@interface PHPickerConfiguration ()

@property (nonatomic, setter=_setAllowsDownscaling:) BOOL _allowsDownscaling;

@end

#endif // USE(APPLE_INTERNAL_SDK)

#endif // HAVE(PHOTOS_UI)
