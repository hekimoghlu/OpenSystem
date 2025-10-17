/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include "pc/audio_rtp_receiver.h"

#include <atomic>

#include "pc/test/mock_voice_media_receive_channel_interface.h"
#include "rtc_base/gunit.h"
#include "rtc_base/thread.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/run_loop.h"

using ::testing::_;
using ::testing::InvokeWithoutArgs;
using ::testing::Mock;

static const int kTimeOut = 100;
static const double kDefaultVolume = 1;
static const double kVolume = 3.7;
static const double kVolumeMuted = 0.0;
static const uint32_t kSsrc = 3;

namespace webrtc {
class AudioRtpReceiverTest : public ::testing::Test {
 protected:
  AudioRtpReceiverTest()
      : worker_(rtc::Thread::Current()),
        receiver_(
            rtc::make_ref_counted<AudioRtpReceiver>(worker_,
                                                    std::string(),
                                                    std::vector<std::string>(),
                                                    false)) {
    EXPECT_CALL(receive_channel_, SetRawAudioSink(kSsrc, _));
    EXPECT_CALL(receive_channel_, SetBaseMinimumPlayoutDelayMs(kSsrc, _));
  }

  ~AudioRtpReceiverTest() {
    EXPECT_CALL(receive_channel_, SetOutputVolume(kSsrc, kVolumeMuted));
    receiver_->SetMediaChannel(nullptr);
  }

  rtc::AutoThread main_thread_;
  rtc::Thread* worker_;
  rtc::scoped_refptr<AudioRtpReceiver> receiver_;
  cricket::MockVoiceMediaReceiveChannelInterface receive_channel_;
};

TEST_F(AudioRtpReceiverTest, SetOutputVolumeIsCalled) {
  std::atomic_int set_volume_calls(0);

  EXPECT_CALL(receive_channel_, SetOutputVolume(kSsrc, kDefaultVolume))
      .WillOnce(InvokeWithoutArgs([&] {
        set_volume_calls++;
        return true;
      }));

  receiver_->track();
  receiver_->track()->set_enabled(true);
  receiver_->SetMediaChannel(&receive_channel_);
  EXPECT_CALL(receive_channel_, SetDefaultRawAudioSink(_)).Times(0);
  receiver_->SetupMediaChannel(kSsrc);

  EXPECT_CALL(receive_channel_, SetOutputVolume(kSsrc, kVolume))
      .WillOnce(InvokeWithoutArgs([&] {
        set_volume_calls++;
        return true;
      }));

  receiver_->OnSetVolume(kVolume);
  EXPECT_TRUE_WAIT(set_volume_calls == 2, kTimeOut);
}

TEST_F(AudioRtpReceiverTest, VolumesSetBeforeStartingAreRespected) {
  // Set the volume before setting the media channel. It should still be used
  // as the initial volume.
  receiver_->OnSetVolume(kVolume);

  receiver_->track()->set_enabled(true);
  receiver_->SetMediaChannel(&receive_channel_);

  // The previosly set initial volume should be propagated to the provided
  // media_channel_ as soon as SetupMediaChannel is called.
  EXPECT_CALL(receive_channel_, SetOutputVolume(kSsrc, kVolume));

  receiver_->SetupMediaChannel(kSsrc);
}

// Tests that OnChanged notifications are processed correctly on the worker
// thread when a media channel pointer is passed to the receiver via the
// constructor.
TEST(AudioRtpReceiver, OnChangedNotificationsAfterConstruction) {
  test::RunLoop loop;
  auto* thread = rtc::Thread::Current();  // Points to loop's thread.
  cricket::MockVoiceMediaReceiveChannelInterface receive_channel;
  auto receiver = rtc::make_ref_counted<AudioRtpReceiver>(
      thread, std::string(), std::vector<std::string>(), true,
      &receive_channel);

  EXPECT_CALL(receive_channel, SetDefaultRawAudioSink(_)).Times(1);
  EXPECT_CALL(receive_channel, SetDefaultOutputVolume(kDefaultVolume)).Times(1);
  receiver->SetupUnsignaledMediaChannel();
  loop.Flush();

  // Mark the track as disabled.
  receiver->track()->set_enabled(false);

  // When the track was marked as disabled, an async notification was queued
  // for the worker thread. This notification should trigger the volume
  // of the media channel to be set to kVolumeMuted.
  // Flush the worker thread, but set the expectation first for the call.
  EXPECT_CALL(receive_channel, SetDefaultOutputVolume(kVolumeMuted)).Times(1);
  loop.Flush();

  EXPECT_CALL(receive_channel, SetDefaultOutputVolume(kVolumeMuted)).Times(1);
  receiver->SetMediaChannel(nullptr);
}

}  // namespace webrtc
