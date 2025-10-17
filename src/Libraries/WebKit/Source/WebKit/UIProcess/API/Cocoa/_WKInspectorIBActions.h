/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#import <WebKit/WKFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@protocol _WKInspectorIBActions <NSObject>

/**
 * @abstract Shows the associated Web Inspector instance.
 */
- (void)show;

/**
 * @abstract Closes the associated Web Inspector instance. This will cause all
 * registered _WKInspectorExtensions to be unregistered and invalidated.
 */
- (void)close;

/**
 * @abstract Opens the Console Tab in the associated Web Inspector instance.
 */
- (void)showConsole;

/**
 * @abstract Opens the Sources Tab in the associated Web Inspector instance.
 */
- (void)showResources;

@end

NS_ASSUME_NONNULL_END
