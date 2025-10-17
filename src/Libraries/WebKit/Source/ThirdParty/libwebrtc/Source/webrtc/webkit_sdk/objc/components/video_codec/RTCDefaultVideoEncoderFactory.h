/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
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

#import "RTCMacros.h"
#import "RTCVideoEncoderFactory.h"

NS_ASSUME_NONNULL_BEGIN

/** This encoder factory include support for all codecs bundled with WebRTC. If using custom
 *  codecs, create custom implementations of RTCVideoEncoderFactory and RTCVideoDecoderFactory.
 */
RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCDefaultVideoEncoderFactory")))
@interface RTCDefaultVideoEncoderFactory : NSObject <RTCVideoEncoderFactory>

- (id)initWithH265:(bool)supportH265 vp9Profile0:(bool)supportsVP9Profile0 vp9Profile2:(bool)supportsVP9Profile2 av1:(bool)supportAv1;
+ (NSArray<RTCVideoCodecInfo *> *)supportedCodecs;
+ (NSArray<RTCVideoCodecInfo *> *)supportedCodecsWithH265:(bool)supportsH265 vp9Profile0:(bool)supportsVP9Profile0 vp9Profile2:(bool)supportsVP9Profile2 av1:(bool)supportsAv1;

@end

NS_ASSUME_NONNULL_END
