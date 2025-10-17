/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

#if ENABLE(APPLE_PAY_AMS_UI)

#include "AppleMediaServicesSPI.h"

#if USE(APPLE_INTERNAL_SDK)

#import <AppleMediaServicesUI/AMSUIEngagementTask.h>

#else // if !USE(APPLE_INTERNAL_SDK)

NS_ASSUME_NONNULL_BEGIN

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS UIViewController;
typedef UIViewController AMSUIViewController;
#else
OBJC_CLASS NSViewController;
typedef NSViewController AMSUIViewController;
#endif

@interface AMSUIEngagementTask : AMSTask <AMSBagConsumer>
- (instancetype)initWithRequest:(AMSEngagementRequest *)request bag:(id<AMSBagProtocol>)bag presentingViewController:(AMSUIViewController *)viewController;
@property (NS_NONATOMIC_IOSONLY, assign) BOOL remotePresentation;
- (AMSPromise<AMSEngagementResult *> *)presentEngagement;
@end

NS_ASSUME_NONNULL_END

#endif // !USE(APPLE_INTERNAL_SDK)

#endif // ENABLE(APPLE_PAY_AMS_UI)
