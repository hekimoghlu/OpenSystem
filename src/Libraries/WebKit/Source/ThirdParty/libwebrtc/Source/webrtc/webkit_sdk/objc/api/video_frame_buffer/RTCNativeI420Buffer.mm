/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 6, 2022.
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
#import "RTCNativeI420Buffer+Private.h"

#include "api/video/i420_buffer.h"

#if !defined(NDEBUG) && defined(WEBRTC_IOS)
#import <UIKit/UIKit.h>
#include "third_party/libyuv/include/libyuv.h"
#endif

@implementation RTCI420Buffer

- (instancetype)initWithWidth:(int)width height:(int)height {
  if (self = [super init]) {
    _i420Buffer = webrtc::I420Buffer::Create(width, height);
  }

  return self;
}

- (instancetype)initWithWidth:(int)width
                       height:(int)height
                        dataY:(const uint8_t *)dataY
                        dataU:(const uint8_t *)dataU
                        dataV:(const uint8_t *)dataV {
  if (self = [super init]) {
    _i420Buffer = webrtc::I420Buffer::Copy(
        width, height, dataY, width, dataU, (width + 1) / 2, dataV, (width + 1) / 2);
  }
  return self;
}

- (instancetype)initWithWidth:(int)width
                       height:(int)height
                      strideY:(int)strideY
                      strideU:(int)strideU
                      strideV:(int)strideV {
  if (self = [super init]) {
    _i420Buffer = webrtc::I420Buffer::Create(width, height, strideY, strideU, strideV);
  }

  return self;
}

- (instancetype)initWithFrameBuffer:(rtc::scoped_refptr<webrtc::I420BufferInterface>)i420Buffer {
  if (self = [super init]) {
    _i420Buffer = i420Buffer;
  }

  return self;
}

- (int)width {
  return _i420Buffer->width();
}

- (int)height {
  return _i420Buffer->height();
}

- (int)strideY {
  return _i420Buffer->StrideY();
}

- (int)strideU {
  return _i420Buffer->StrideU();
}

- (int)strideV {
  return _i420Buffer->StrideV();
}

- (int)chromaWidth {
  return _i420Buffer->ChromaWidth();
}

- (int)chromaHeight {
  return _i420Buffer->ChromaHeight();
}

- (const uint8_t *)dataY {
  return _i420Buffer->DataY();
}

- (const uint8_t *)dataU {
  return _i420Buffer->DataU();
}

- (const uint8_t *)dataV {
  return _i420Buffer->DataV();
}

- (id<RTCI420Buffer>)toI420 {
  return self;
}

#pragma mark - Private

- (rtc::scoped_refptr<webrtc::I420BufferInterface>)nativeI420Buffer {
  return _i420Buffer;
}

#if defined(WEBRTC_WEBKIT_BUILD)
- (void)close {
  _i420Buffer = nullptr;
}
#endif

#pragma mark - Debugging

#if !defined(NDEBUG) && defined(WEBRTC_IOS)
- (id)debugQuickLookObject {
  UIGraphicsBeginImageContext(CGSizeMake(_i420Buffer->width(), _i420Buffer->height()));
  CGContextRef c = UIGraphicsGetCurrentContext();
  uint8_t *ctxData = (uint8_t *)CGBitmapContextGetData(c);

  libyuv::I420ToARGB(_i420Buffer->DataY(),
                     _i420Buffer->StrideY(),
                     _i420Buffer->DataU(),
                     _i420Buffer->StrideU(),
                     _i420Buffer->DataV(),
                     _i420Buffer->StrideV(),
                     ctxData,
                     CGBitmapContextGetBytesPerRow(c),
                     CGBitmapContextGetWidth(c),
                     CGBitmapContextGetHeight(c));

  UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
  UIGraphicsEndImageContext();

  return image;
}
#endif

@end
