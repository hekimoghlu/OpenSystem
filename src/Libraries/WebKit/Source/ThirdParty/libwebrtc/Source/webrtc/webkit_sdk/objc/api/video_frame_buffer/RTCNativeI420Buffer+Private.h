/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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
#import "RTCNativeI420Buffer.h"

#include "api/video/i420_buffer.h"

NS_ASSUME_NONNULL_BEGIN

@interface RTCI420Buffer () {
 @protected
  rtc::scoped_refptr<webrtc::I420BufferInterface> _i420Buffer;
}

/** Initialize an RTCI420Buffer with its backing I420BufferInterface. */
- (instancetype)initWithFrameBuffer:(rtc::scoped_refptr<webrtc::I420BufferInterface>)i420Buffer;
- (rtc::scoped_refptr<webrtc::I420BufferInterface>)nativeI420Buffer;

#if defined(WEBRTC_WEBKIT_BUILD)
- (void)close;
#endif
@end

NS_ASSUME_NONNULL_END
