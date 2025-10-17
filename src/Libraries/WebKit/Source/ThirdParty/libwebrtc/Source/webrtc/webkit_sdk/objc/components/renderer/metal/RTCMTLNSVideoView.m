/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#import "RTCMTLNSVideoView.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "base/RTCVideoFrame.h"

#import "RTCMTLI420Renderer.h"

@interface RTCMTLNSVideoView ()<MTKViewDelegate>
@property(nonatomic) id<RTCMTLRenderer> renderer;
@property(nonatomic, strong) MTKView *metalView;
@property(atomic, strong) RTCVideoFrame *videoFrame;
@end

@implementation RTCMTLNSVideoView {
  id<RTCMTLRenderer> _renderer;
}

@synthesize delegate = _delegate;
@synthesize renderer = _renderer;
@synthesize metalView = _metalView;
@synthesize videoFrame = _videoFrame;

- (instancetype)initWithFrame:(CGRect)frameRect {
  self = [super initWithFrame:frameRect];
  if (self) {
    [self configure];
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder *)aCoder {
  self = [super initWithCoder:aCoder];
  if (self) {
    [self configure];
  }
  return self;
}

#pragma mark - Private

+ (BOOL)isMetalAvailable {
  return [MTLCopyAllDevices() count] > 0;
}

- (void)configure {
  if ([[self class] isMetalAvailable]) {
    _metalView = [[MTKView alloc] initWithFrame:self.bounds];
    [self addSubview:_metalView];
    _metalView.layerContentsPlacement = NSViewLayerContentsPlacementScaleProportionallyToFit;
    _metalView.translatesAutoresizingMaskIntoConstraints = NO;
    _metalView.framebufferOnly = YES;
    _metalView.delegate = self;

    _renderer = [[RTCMTLI420Renderer alloc] init];
    if (![(RTCMTLI420Renderer *)_renderer addRenderingDestination:_metalView]) {
      _renderer = nil;
    };
  }
}

- (void)updateConstraints {
  NSDictionary *views = NSDictionaryOfVariableBindings(_metalView);

  NSArray *constraintsHorizontal =
      [NSLayoutConstraint constraintsWithVisualFormat:@"H:|-0-[_metalView]-0-|"
                                              options:0
                                              metrics:nil
                                                views:views];
  [self addConstraints:constraintsHorizontal];

  NSArray *constraintsVertical =
      [NSLayoutConstraint constraintsWithVisualFormat:@"V:|-0-[_metalView]-0-|"
                                              options:0
                                              metrics:nil
                                                views:views];
  [self addConstraints:constraintsVertical];
  [super updateConstraints];
}

#pragma mark - MTKViewDelegate methods
- (void)drawInMTKView:(nonnull MTKView *)view {
  if (self.videoFrame == nil) {
    return;
  }
  if (view == self.metalView) {
    [_renderer drawFrame:self.videoFrame];
  }
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
}

#pragma mark - RTCVideoRenderer

- (void)setSize:(CGSize)size {
  _metalView.drawableSize = size;
  dispatch_async(dispatch_get_main_queue(), ^{
    [self.delegate videoView:self didChangeVideoSize:size];
  });
  [_metalView draw];
}

- (void)renderFrame:(nullable RTCVideoFrame *)frame {
  if (frame == nil) {
    return;
  }
  self.videoFrame = [frame newI420VideoFrame];
}

@end
