/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#import "RTCMTLRGBRenderer.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "RTCMTLRenderer+Private.h"
#import "base/RTCLogging.h"
#import "base/RTCVideoFrame.h"
#import "base/RTCVideoFrameBuffer.h"
#import "components/video_frame_buffer/RTCCVPixelBuffer.h"

#include "rtc_base/checks.h"

static NSString *const shaderSource = MTL_STRINGIFY(
    using namespace metal;

    typedef struct {
      packed_float2 position;
      packed_float2 texcoord;
    } Vertex;

    typedef struct {
      float4 position[[position]];
      float2 texcoord;
    } VertexIO;

    vertex VertexIO vertexPassthrough(constant Vertex *verticies[[buffer(0)]],
                                      uint vid[[vertex_id]]) {
      VertexIO out;
      constant Vertex &v = verticies[vid];
      out.position = float4(float2(v.position), 0.0, 1.0);
      out.texcoord = v.texcoord;
      return out;
    }

    fragment half4 fragmentColorConversion(VertexIO in[[stage_in]],
                                           texture2d<half, access::sample> texture[[texture(0)]],
                                           constant bool &isARGB[[buffer(0)]]) {
      constexpr sampler s(address::clamp_to_edge, filter::linear);

      half4 out = texture.sample(s, in.texcoord);
      if (isARGB) {
        out = half4(out.g, out.b, out.a, out.r);
      }

      return out;
    });

@implementation RTCMTLRGBRenderer {
  // Textures.
  CVMetalTextureCacheRef _textureCache;
  id<MTLTexture> _texture;

  // Uniforms.
  id<MTLBuffer> _uniformsBuffer;
}

- (BOOL)addRenderingDestination:(__kindof MTKView *)view {
  if ([super addRenderingDestination:view]) {
    return [self initializeTextureCache];
  }
  return NO;
}

- (BOOL)initializeTextureCache {
  CVReturn status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, [self currentMetalDevice],
                                              nil, &_textureCache);
  if (status != kCVReturnSuccess) {
    RTCLogError(@"Metal: Failed to initialize metal texture cache. Return status is %d", status);
    return NO;
  }

  return YES;
}

- (NSString *)shaderSource {
  return shaderSource;
}

- (void)getWidth:(nonnull int *)width
          height:(nonnull int *)height
       cropWidth:(nonnull int *)cropWidth
      cropHeight:(nonnull int *)cropHeight
           cropX:(nonnull int *)cropX
           cropY:(nonnull int *)cropY
         ofFrame:(nonnull RTCVideoFrame *)frame {
  RTCCVPixelBuffer *pixelBuffer = (RTCCVPixelBuffer *)frame.buffer;
  *width = CVPixelBufferGetWidth(pixelBuffer.pixelBuffer);
  *height = CVPixelBufferGetHeight(pixelBuffer.pixelBuffer);
  *cropWidth = pixelBuffer.cropWidth;
  *cropHeight = pixelBuffer.cropHeight;
  *cropX = pixelBuffer.cropX;
  *cropY = pixelBuffer.cropY;
}

- (BOOL)setupTexturesForFrame:(nonnull RTCVideoFrame *)frame {
  RTC_DCHECK([frame.buffer isKindOfClass:[RTCCVPixelBuffer class]]);
  if (![super setupTexturesForFrame:frame]) {
    return NO;
  }
  CVPixelBufferRef pixelBuffer = ((RTCCVPixelBuffer *)frame.buffer).pixelBuffer;

  id<MTLTexture> gpuTexture = nil;
  CVMetalTextureRef textureOut = nullptr;
  bool isARGB;

  int width = CVPixelBufferGetWidth(pixelBuffer);
  int height = CVPixelBufferGetHeight(pixelBuffer);
  OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);

  MTLPixelFormat mtlPixelFormat;
  if (pixelFormat == kCVPixelFormatType_32BGRA) {
    mtlPixelFormat = MTLPixelFormatBGRA8Unorm;
    isARGB = false;
  } else if (pixelFormat == kCVPixelFormatType_32ARGB) {
    mtlPixelFormat = MTLPixelFormatRGBA8Unorm;
    isARGB = true;
  } else {
    RTC_NOTREACHED();
    return NO;
  }

  CVReturn result = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, _textureCache, pixelBuffer, nil, mtlPixelFormat,
                width, height, 0, &textureOut);
  if (result == kCVReturnSuccess) {
    gpuTexture = CVMetalTextureGetTexture(textureOut);
  }
  CVBufferRelease(textureOut);

  if (gpuTexture != nil) {
    _texture = gpuTexture;
    _uniformsBuffer =
        [[self currentMetalDevice] newBufferWithBytes:&isARGB
                                               length:sizeof(isARGB)
                                              options:MTLResourceCPUCacheModeDefaultCache];
    return YES;
  }

  return NO;
}

- (void)uploadTexturesToRenderEncoder:(id<MTLRenderCommandEncoder>)renderEncoder {
  [renderEncoder setFragmentTexture:_texture atIndex:0];
  [renderEncoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:0];
}

- (void)dealloc {
  if (_textureCache) {
    CFRelease(_textureCache);
  }
}

@end
