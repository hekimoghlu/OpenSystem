/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#import "RTCDispatcher+Private.h"

static dispatch_queue_t kAudioSessionQueue = nil;
static dispatch_queue_t kCaptureSessionQueue = nil;

@implementation RTCDispatcher

+ (void)initialize {
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    kAudioSessionQueue = dispatch_queue_create(
        "org.webrtc.RTCDispatcherAudioSession",
        DISPATCH_QUEUE_SERIAL);
    kCaptureSessionQueue = dispatch_queue_create(
        "org.webrtc.RTCDispatcherCaptureSession",
        DISPATCH_QUEUE_SERIAL);
  });
}

+ (void)dispatchAsyncOnType:(RTCDispatcherQueueType)dispatchType
                      block:(dispatch_block_t)block {
  dispatch_queue_t queue = [self dispatchQueueForType:dispatchType];
  dispatch_async(queue, block);
}

+ (BOOL)isOnQueueForType:(RTCDispatcherQueueType)dispatchType {
  dispatch_queue_t targetQueue = [self dispatchQueueForType:dispatchType];
  const char* targetLabel = dispatch_queue_get_label(targetQueue);
  const char* currentLabel = dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL);

  NSAssert(strlen(targetLabel) > 0, @"Label is required for the target queue.");
  NSAssert(strlen(currentLabel) > 0, @"Label is required for the current queue.");

  return strcmp(targetLabel, currentLabel) == 0;
}

#pragma mark - Private

+ (dispatch_queue_t)dispatchQueueForType:(RTCDispatcherQueueType)dispatchType {
  switch (dispatchType) {
    case RTCDispatcherTypeMain:
      return dispatch_get_main_queue();
    case RTCDispatcherTypeCaptureSession:
      return kCaptureSessionQueue;
    case RTCDispatcherTypeAudioSession:
      return kAudioSessionQueue;
  }
}

@end
