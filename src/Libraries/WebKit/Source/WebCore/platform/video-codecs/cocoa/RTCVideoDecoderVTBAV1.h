/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#if USE(LIBWEBRTC)

typedef struct __CVBuffer* CVPixelBufferRef;

using RTCVideoDecoderVTBAV1Callback = void (^)(CVPixelBufferRef, int64_t timeStamp, int64_t timeStampNs, bool);

__attribute__((objc_runtime_name("Web_RTCVideoDecoderVTBAV1")))
@interface RTCVideoDecoderVTBAV1 : NSObject
- (void)setCallback:(RTCVideoDecoderVTBAV1Callback)callback;
- (void)setWidth:(uint16_t)width height:(uint16_t)height;
- (NSInteger)releaseDecoder;
- (NSInteger)decodeData:(const uint8_t *)data size:(size_t)size timeStamp:(int64_t)timeStamp;
- (void)flush;
@end

#endif // USE(LIBWEBRTC)
