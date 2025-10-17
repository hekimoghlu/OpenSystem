/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#if ENABLE(APP_HIGHLIGHTS)

#if USE(APPLE_INTERNAL_SDK)
#import <Synapse/SYNotesActivationObserver.h>
#else

NS_ASSUME_NONNULL_BEGIN

typedef void(^SYNotesActivationObserverHandler)(BOOL isVisible);

@interface SYNotesActivationObserver : NSObject

@property (nonatomic, readonly, getter=isVisible) BOOL visible;

@property (nonatomic, readonly) CGRect visibleFrame;

- (instancetype)initWithHandler:(nullable SYNotesActivationObserverHandler)handler;

@end

NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)

#endif // ENABLE(APP_HIGHLIGHTS)
