/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#import "RTCVideoFrame.h"

#import "RTCI420Buffer.h"
#import "RTCVideoFrameBuffer.h"

@implementation RTCVideoFrame {
  RTCVideoRotation _rotation;
  int64_t _timeStampNs;
  uint64_t _duration;
}

@synthesize buffer = _buffer;
@synthesize timeStamp;

- (int)width {
  return _buffer.width;
}

- (int)height {
  return _buffer.height;
}

- (RTCVideoRotation)rotation {
  return _rotation;
}

- (int64_t)timeStampNs {
  return _timeStampNs;
}

- (uint64_t)duration {
  return _duration;
}

- (RTCVideoFrame *)newI420VideoFrame {
  return [[RTCVideoFrame alloc] initWithBuffer:[_buffer toI420]
                                      rotation:_rotation
                                   timeStampNs:_timeStampNs];
}

- (instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                           rotation:(RTCVideoRotation)rotation
                        timeStampNs:(int64_t)timeStampNs {
  // Deprecated.
  return nil;
}

- (instancetype)initWithPixelBuffer:(CVPixelBufferRef)pixelBuffer
                        scaledWidth:(int)scaledWidth
                       scaledHeight:(int)scaledHeight
                          cropWidth:(int)cropWidth
                         cropHeight:(int)cropHeight
                              cropX:(int)cropX
                              cropY:(int)cropY
                           rotation:(RTCVideoRotation)rotation
                        timeStampNs:(int64_t)timeStampNs {
  // Deprecated.
  return nil;
}

- (instancetype)initWithBuffer:(id<RTCVideoFrameBuffer>)buffer
                      rotation:(RTCVideoRotation)rotation
                   timeStampNs:(int64_t)timeStampNs {
  if (self = [super init]) {
    _buffer = buffer;
    _rotation = rotation;
    _timeStampNs = timeStampNs;
    _duration = 0;
  }

  return self;
}

@end
