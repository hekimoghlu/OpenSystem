/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#import <Foundation/Foundation.h>

#import "RTCMacros.h"

typedef NS_ENUM(NSInteger, RTCDispatcherQueueType) {
  // Main dispatcher queue.
  RTCDispatcherTypeMain,
  // Used for starting/stopping AVCaptureSession, and assigning
  // capture session to AVCaptureVideoPreviewLayer.
  RTCDispatcherTypeCaptureSession,
  // Used for operations on AVAudioSession.
  RTCDispatcherTypeAudioSession,
};

/** Dispatcher that asynchronously dispatches blocks to a specific
 *  shared dispatch queue.
 */
RTC_OBJC_EXPORT
@interface RTCDispatcher : NSObject

- (instancetype)init NS_UNAVAILABLE;

/** Dispatch the block asynchronously on the queue for dispatchType.
 *  @param dispatchType The queue type to dispatch on.
 *  @param block The block to dispatch asynchronously.
 */
+ (void)dispatchAsyncOnType:(RTCDispatcherQueueType)dispatchType block:(dispatch_block_t)block;

/** Returns YES if run on queue for the dispatchType otherwise NO.
 *  Useful for asserting that a method is run on a correct queue.
 */
+ (BOOL)isOnQueueForType:(RTCDispatcherQueueType)dispatchType;

@end
