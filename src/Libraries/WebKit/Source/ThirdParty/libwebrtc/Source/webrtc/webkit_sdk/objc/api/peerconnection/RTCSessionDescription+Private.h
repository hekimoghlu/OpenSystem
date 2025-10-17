/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#import "RTCSessionDescription.h"

#include "api/jsep.h"

NS_ASSUME_NONNULL_BEGIN

@interface RTCSessionDescription ()

/**
 * The native SessionDescriptionInterface representation of this
 * RTCSessionDescription object. This is needed to pass to the underlying C++
 * APIs.
 */
@property(nonatomic, readonly, nullable) webrtc::SessionDescriptionInterface *nativeDescription;

/**
 * Initialize an RTCSessionDescription from a native
 * SessionDescriptionInterface. No ownership is taken of the native session
 * description.
 */
- (instancetype)initWithNativeDescription:
        (const webrtc::SessionDescriptionInterface *)nativeDescription;

+ (std::string)stdStringForType:(RTCSdpType)type;

+ (RTCSdpType)typeForStdString:(const std::string &)string;

@end

NS_ASSUME_NONNULL_END
