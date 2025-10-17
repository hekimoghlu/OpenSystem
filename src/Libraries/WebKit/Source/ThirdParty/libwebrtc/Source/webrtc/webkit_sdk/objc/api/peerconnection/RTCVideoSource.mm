/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#import "RTCVideoSource+Private.h"

#include "api/video_track_source_proxy.h"
#include "rtc_base/checks.h"
#include "sdk/objc/native/src/objc_video_track_source.h"

static webrtc::ObjCVideoTrackSource *getObjCVideoSource(
    const rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> nativeSource) {
  webrtc::VideoTrackSourceProxy *proxy_source =
      static_cast<webrtc::VideoTrackSourceProxy *>(nativeSource.get());
  return static_cast<webrtc::ObjCVideoTrackSource *>(proxy_source->internal());
}

// TODO(magjed): Refactor this class and target ObjCVideoTrackSource only once
// RTCAVFoundationVideoSource is gone. See http://crbug/webrtc/7177 for more
// info.
@implementation RTCVideoSource {
  rtc::scoped_refptr<webrtc::VideoTrackSourceInterface> _nativeVideoSource;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
              nativeVideoSource:
                  (rtc::scoped_refptr<webrtc::VideoTrackSourceInterface>)nativeVideoSource {
  RTC_DCHECK(factory);
  RTC_DCHECK(nativeVideoSource);
  if (self = [super initWithFactory:factory
                  nativeMediaSource:nativeVideoSource
                               type:RTCMediaSourceTypeVideo]) {
    _nativeVideoSource = nativeVideoSource;
  }
  return self;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
              nativeMediaSource:(rtc::scoped_refptr<webrtc::MediaSourceInterface>)nativeMediaSource
                           type:(RTCMediaSourceType)type {
  RTC_NOTREACHED();
  return nil;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
                signalingThread:(rtc::Thread *)signalingThread
                   workerThread:(rtc::Thread *)workerThread {
  rtc::scoped_refptr<webrtc::ObjCVideoTrackSource> objCVideoTrackSource(
      new rtc::RefCountedObject<webrtc::ObjCVideoTrackSource>());

  return [self initWithFactory:factory
             nativeVideoSource:webrtc::VideoTrackSourceProxy::Create(
                                   signalingThread, workerThread, objCVideoTrackSource)];
}

- (NSString *)description {
  NSString *stateString = [[self class] stringForState:self.state];
  return [NSString stringWithFormat:@"RTCVideoSource( %p ): %@", self, stateString];
}

- (void)capturer:(RTCVideoCapturer *)capturer didCaptureVideoFrame:(RTCVideoFrame *)frame {
  getObjCVideoSource(_nativeVideoSource)->OnCapturedFrame(frame);
}

- (void)adaptOutputFormatToWidth:(int)width height:(int)height fps:(int)fps {
  getObjCVideoSource(_nativeVideoSource)->OnOutputFormatRequest(width, height, fps);
}

#pragma mark - Private

- (rtc::scoped_refptr<webrtc::VideoTrackSourceInterface>)nativeVideoSource {
  return _nativeVideoSource;
}

@end
