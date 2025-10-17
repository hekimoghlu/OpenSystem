/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#import "RTCCameraPreviewView.h"

#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

#import "RTCDispatcher+Private.h"

@implementation RTCCameraPreviewView

@synthesize captureSession = _captureSession;

+ (Class)layerClass {
  return [AVCaptureVideoPreviewLayer class];
}

- (instancetype)initWithFrame:(CGRect)aRect {
  self = [super initWithFrame:aRect];
  if (self) {
    [self addOrientationObserver];
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)aDecoder {
  self = [super initWithCoder:aDecoder];
  if (self) {
    [self addOrientationObserver];
  }
  return self;
}

- (void)dealloc {
  [self removeOrientationObserver];
}

- (void)setCaptureSession:(AVCaptureSession *)captureSession {
  if (_captureSession == captureSession) {
    return;
  }
  _captureSession = captureSession;
  [RTCDispatcher
      dispatchAsyncOnType:RTCDispatcherTypeMain
                    block:^{
                      AVCaptureVideoPreviewLayer *previewLayer = [self previewLayer];
                      [RTCDispatcher
                          dispatchAsyncOnType:RTCDispatcherTypeCaptureSession
                                        block:^{
                                          previewLayer.session = captureSession;
                                          [RTCDispatcher
                                              dispatchAsyncOnType:RTCDispatcherTypeMain
                                                            block:^{
                                                              [self setCorrectVideoOrientation];
                                                            }];
                                        }];
                    }];
}

- (void)layoutSubviews {
  [super layoutSubviews];

  // Update the video orientation based on the device orientation.
  [self setCorrectVideoOrientation];
}

-(void)orientationChanged:(NSNotification *)notification {
  [self setCorrectVideoOrientation];
}

- (void)setCorrectVideoOrientation {
  // Get current device orientation.
  UIDeviceOrientation deviceOrientation = [UIDevice currentDevice].orientation;
  AVCaptureVideoPreviewLayer *previewLayer = [self previewLayer];

  // First check if we are allowed to set the video orientation.
  if (previewLayer.connection.isVideoOrientationSupported) {
    // Set the video orientation based on device orientation.
    if (deviceOrientation == UIInterfaceOrientationPortraitUpsideDown) {
      previewLayer.connection.videoOrientation =
          AVCaptureVideoOrientationPortraitUpsideDown;
    } else if (deviceOrientation == UIInterfaceOrientationLandscapeRight) {
      previewLayer.connection.videoOrientation =
          AVCaptureVideoOrientationLandscapeRight;
    } else if (deviceOrientation == UIInterfaceOrientationLandscapeLeft) {
      previewLayer.connection.videoOrientation =
          AVCaptureVideoOrientationLandscapeLeft;
    } else if (deviceOrientation == UIInterfaceOrientationPortrait) {
      previewLayer.connection.videoOrientation =
          AVCaptureVideoOrientationPortrait;
    }
    // If device orientation switches to FaceUp or FaceDown, don't change video orientation.
  }
}

#pragma mark - Private

- (void)addOrientationObserver {
  [[NSNotificationCenter defaultCenter] addObserver:self
                                            selector:@selector(orientationChanged:)
                                                name:UIDeviceOrientationDidChangeNotification
                                              object:nil];
}

- (void)removeOrientationObserver {
  [[NSNotificationCenter defaultCenter] removeObserver:self
                                                  name:UIDeviceOrientationDidChangeNotification
                                                object:nil];
}

- (AVCaptureVideoPreviewLayer *)previewLayer {
  return (AVCaptureVideoPreviewLayer *)self.layer;
}

@end
