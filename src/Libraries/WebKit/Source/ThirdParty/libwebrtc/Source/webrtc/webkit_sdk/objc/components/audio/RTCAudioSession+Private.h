/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#import "RTCAudioSession.h"

NS_ASSUME_NONNULL_BEGIN

@class RTCAudioSessionConfiguration;

@interface RTCAudioSession ()

/** Number of times setActive:YES has succeeded without a balanced call to
 *  setActive:NO.
 */
@property(nonatomic, readonly) int activationCount;

/** The number of times |beginWebRTCSession| was called without a balanced call
 *  to |endWebRTCSession|.
 */
@property(nonatomic, readonly) int webRTCSessionCount;

/** Convenience BOOL that checks useManualAudio and isAudioEnebled. */
@property(readonly) BOOL canPlayOrRecord;

/** Tracks whether we have been sent an interruption event that hasn't been matched by either an
 *  interrupted end event or a foreground event.
 */
@property(nonatomic, assign) BOOL isInterrupted;

- (BOOL)checkLock:(NSError **)outError;

/** Adds the delegate to the list of delegates, and places it at the front of
 *  the list. This delegate will be notified before other delegates of
 *  audio events.
 */
- (void)pushDelegate:(id<RTCAudioSessionDelegate>)delegate;

/** Signals RTCAudioSession that a WebRTC session is about to begin and
 *  audio configuration is needed. Will configure the audio session for WebRTC
 *  if not already configured and if configuration is not delayed.
 *  Successful calls must be balanced by a call to endWebRTCSession.
 */
- (BOOL)beginWebRTCSession:(NSError **)outError;

/** Signals RTCAudioSession that a WebRTC session is about to end and audio
 *  unconfiguration is needed. Will unconfigure the audio session for WebRTC
 *  if this is the last unmatched call and if configuration is not delayed.
 */
- (BOOL)endWebRTCSession:(NSError **)outError;

/** Configure the audio session for WebRTC. This call will fail if the session
 *  is already configured. On other failures, we will attempt to restore the
 *  previously used audio session configuration.
 *  |lockForConfiguration| must be called first.
 *  Successful calls to configureWebRTCSession must be matched by calls to
 *  |unconfigureWebRTCSession|.
 */
- (BOOL)configureWebRTCSession:(NSError **)outError;

/** Unconfigures the session for WebRTC. This will attempt to restore the
 *  audio session to the settings used before |configureWebRTCSession| was
 *  called.
 *  |lockForConfiguration| must be called first.
 */
- (BOOL)unconfigureWebRTCSession:(NSError **)outError;

/** Returns a configuration error with the given description. */
- (NSError *)configurationErrorWithDescription:(NSString *)description;

// Properties and methods for tests.
- (void)notifyDidBeginInterruption;
- (void)notifyDidEndInterruptionWithShouldResumeSession:(BOOL)shouldResumeSession;
- (void)notifyDidChangeRouteWithReason:(AVAudioSessionRouteChangeReason)reason
                         previousRoute:(AVAudioSessionRouteDescription *)previousRoute;
- (void)notifyMediaServicesWereLost;
- (void)notifyMediaServicesWereReset;
- (void)notifyDidChangeCanPlayOrRecord:(BOOL)canPlayOrRecord;
- (void)notifyDidStartPlayOrRecord;
- (void)notifyDidStopPlayOrRecord;
- (void)notifyDidDetectPlayoutGlitch:(int64_t)totalNumberOfGlitches;

@end

NS_ASSUME_NONNULL_END
