/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
#import <Foundation/Foundation.h>

#if defined(TARGET_OS_VISION) && TARGET_OS_VISION

NS_ASSUME_NONNULL_BEGIN

@class WKSPreviewWindowController;

@protocol WKSPreviewWindowControllerDelegate <NSObject>
- (void)previewWindowControllerDidClose:(id)previewWindowController;
@end

@interface WKSPreviewWindowController : NSObject
@property (nonatomic, weak, nullable) id <WKSPreviewWindowControllerDelegate> delegate;

- (instancetype)initWithURL:(NSURL *)url sceneID:(NSString *)sceneID NS_DESIGNATED_INITIALIZER;
- (void)presentWindow;
- (void)updateImage:(NSURL *)url NS_SWIFT_NAME(updateImage(url:));
- (void)dismissWindow;
@end

NS_ASSUME_NONNULL_END

#endif
