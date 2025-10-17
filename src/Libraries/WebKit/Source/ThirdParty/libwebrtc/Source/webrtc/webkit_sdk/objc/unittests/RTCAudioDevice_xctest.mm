/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#import <XCTest/XCTest.h>

#include "api/task_queue/default_task_queue_factory.h"

#import "sdk/objc/components/audio/RTCAudioSession+Private.h"
#import "sdk/objc/native/api/audio_device_module.h"
#import "sdk/objc/native/src/audio/audio_device_ios.h"

@interface RTCAudioDeviceTests : XCTestCase {
  rtc::scoped_refptr<webrtc::AudioDeviceModule> _audioDeviceModule;
  std::unique_ptr<webrtc::ios_adm::AudioDeviceIOS> _audio_device;
}

@property(nonatomic) RTCAudioSession *audioSession;

@end

@implementation RTCAudioDeviceTests

@synthesize audioSession = _audioSession;

- (void)setUp {
  [super setUp];

  _audioDeviceModule = webrtc::CreateAudioDeviceModule();
  _audio_device.reset(new webrtc::ios_adm::AudioDeviceIOS());
  self.audioSession = [RTCAudioSession sharedInstance];

  NSError *error = nil;
  [self.audioSession lockForConfiguration];
  [self.audioSession setCategory:AVAudioSessionCategoryPlayAndRecord withOptions:0 error:&error];
  XCTAssertNil(error);

  [self.audioSession setMode:AVAudioSessionModeVoiceChat error:&error];
  XCTAssertNil(error);

  [self.audioSession setActive:YES error:&error];
  XCTAssertNil(error);

  [self.audioSession unlockForConfiguration];
}

- (void)tearDown {
  _audio_device->Terminate();
  _audio_device.reset(nullptr);
  _audioDeviceModule = nullptr;
  [self.audioSession notifyDidEndInterruptionWithShouldResumeSession:NO];

  [super tearDown];
}

// Verifies that the AudioDeviceIOS is_interrupted_ flag is reset correctly
// after an iOS AVAudioSessionInterruptionTypeEnded notification event.
// AudioDeviceIOS listens to RTCAudioSession interrupted notifications by:
// - In AudioDeviceIOS.InitPlayOrRecord registers its audio_session_observer_
//   callback with RTCAudioSession's delegate list.
// - When RTCAudioSession receives an iOS audio interrupted notification, it
//   passes the notification to callbacks in its delegate list which sets
//   AudioDeviceIOS's is_interrupted_ flag to true.
// - When AudioDeviceIOS.ShutdownPlayOrRecord is called, its
//   audio_session_observer_ callback is removed from RTCAudioSessions's
//   delegate list.
//   So if RTCAudioSession receives an iOS end audio interruption notification,
//   AudioDeviceIOS is not notified as its callback is not in RTCAudioSession's
//   delegate list. This causes AudioDeviceIOS's is_interrupted_ flag to be in
//   the wrong (true) state and the audio session will ignore audio changes.
// As RTCAudioSession keeps its own interrupted state, the fix is to initialize
// AudioDeviceIOS's is_interrupted_ flag to RTCAudioSession's isInterrupted
// flag in AudioDeviceIOS.InitPlayOrRecord.
- (void)testInterruptedAudioSession {
  XCTAssertTrue(self.audioSession.isActive);
  XCTAssertTrue([self.audioSession.category isEqual:AVAudioSessionCategoryPlayAndRecord] ||
                [self.audioSession.category isEqual:AVAudioSessionCategoryPlayback]);
  XCTAssertEqual(AVAudioSessionModeVoiceChat, self.audioSession.mode);

  std::unique_ptr<webrtc::TaskQueueFactory> task_queue_factory =
      webrtc::CreateDefaultTaskQueueFactory();
  std::unique_ptr<webrtc::AudioDeviceBuffer> audio_buffer;
  audio_buffer.reset(new webrtc::AudioDeviceBuffer(task_queue_factory.get()));
  _audio_device->AttachAudioBuffer(audio_buffer.get());
  XCTAssertEqual(webrtc::AudioDeviceGeneric::InitStatus::OK, _audio_device->Init());
  XCTAssertEqual(0, _audio_device->InitPlayout());
  XCTAssertEqual(0, _audio_device->StartPlayout());

  // Force interruption.
  [self.audioSession notifyDidBeginInterruption];

  // Wait for notification to propagate.
  rtc::ThreadManager::ProcessAllMessageQueuesForTesting();
  XCTAssertTrue(_audio_device->IsInterrupted());

  // Force it for testing.
  _audio_device->StopPlayout();

  [self.audioSession notifyDidEndInterruptionWithShouldResumeSession:YES];
  // Wait for notification to propagate.
  rtc::ThreadManager::ProcessAllMessageQueuesForTesting();
  XCTAssertTrue(_audio_device->IsInterrupted());

  _audio_device->Init();
  _audio_device->InitPlayout();
  XCTAssertFalse(_audio_device->IsInterrupted());
}

@end
