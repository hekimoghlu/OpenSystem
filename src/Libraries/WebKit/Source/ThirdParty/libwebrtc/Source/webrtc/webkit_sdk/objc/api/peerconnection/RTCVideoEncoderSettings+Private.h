/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#import "base/RTCVideoEncoderSettings.h"

#include "modules/video_coding/include/video_codec_interface.h"

NS_ASSUME_NONNULL_BEGIN

/* Interfaces for converting to/from internal C++ formats. */
@interface RTCVideoEncoderSettings (Private)

- (instancetype)initWithNativeVideoCodec:(const webrtc::VideoCodec *__nullable)videoCodec;
- (webrtc::VideoCodec)nativeVideoCodec;

@end

NS_ASSUME_NONNULL_END
