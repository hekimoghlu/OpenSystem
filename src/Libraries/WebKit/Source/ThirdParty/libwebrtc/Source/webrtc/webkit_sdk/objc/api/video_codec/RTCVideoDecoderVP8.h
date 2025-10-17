/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
#import "RTCVideoDecoder.h"

RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCVideoDecoderVP8")))
@interface RTCVideoDecoderVP8 : NSObject

/* This returns a VP8 decoder that can be returned from a RTCVideoDecoderFactory injected into
 * RTCPeerConnectionFactory. Even though it implements the RTCVideoDecoder protocol, it can not be
 * used independently from the RTCPeerConnectionFactory.
 */
+ (id<RTCVideoDecoder>)vp8Decoder;

@end
