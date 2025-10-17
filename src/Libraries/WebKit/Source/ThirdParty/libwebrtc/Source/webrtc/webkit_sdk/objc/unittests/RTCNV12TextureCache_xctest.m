/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <GLKit/GLKit.h>
#import <XCTest/XCTest.h>

#import "base/RTCVideoFrame.h"
#import "base/RTCVideoFrameBuffer.h"
#import "components/renderer/opengl/RTCNV12TextureCache.h"
#import "components/video_frame_buffer/RTCCVPixelBuffer.h"

@interface RTCNV12TextureCacheTests : XCTestCase
@end

@implementation RTCNV12TextureCacheTests {
  EAGLContext *_glContext;
  RTCNV12TextureCache *_nv12TextureCache;
}

- (void)setUp {
  [super setUp];
  _glContext = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES3];
  if (!_glContext) {
    _glContext = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
  }
  _nv12TextureCache = [[RTCNV12TextureCache alloc] initWithContext:_glContext];
}

- (void)tearDown {
  _nv12TextureCache = nil;
  _glContext = nil;
  [super tearDown];
}

- (void)testNV12TextureCacheDoesNotCrashOnEmptyFrame {
  CVPixelBufferRef nullPixelBuffer = NULL;
  RTCCVPixelBuffer *badFrameBuffer = [[RTCCVPixelBuffer alloc] initWithPixelBuffer:nullPixelBuffer];
  RTCVideoFrame *badFrame = [[RTCVideoFrame alloc] initWithBuffer:badFrameBuffer
                                                         rotation:RTCVideoRotation_0
                                                      timeStampNs:0];
  [_nv12TextureCache uploadFrameToTextures:badFrame];
}

@end
