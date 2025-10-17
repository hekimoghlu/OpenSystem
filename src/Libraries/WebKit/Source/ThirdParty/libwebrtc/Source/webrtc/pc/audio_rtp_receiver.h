/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#ifndef PC_AUDIO_RTP_RECEIVER_H_
#define PC_AUDIO_RTP_RECEIVER_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "api/crypto/frame_decryptor_interface.h"
#include "api/dtls_transport_interface.h"
#include "api/frame_transformer_interface.h"
#include "api/media_stream_interface.h"
#include "api/media_types.h"
#include "api/rtp_parameters.h"
#include "api/rtp_receiver_interface.h"
#include "api/scoped_refptr.h"
#include "api/sequence_checker.h"
#include "api/task_queue/pending_task_safety_flag.h"
#include "api/transport/rtp/rtp_source.h"
#include "media/base/media_channel.h"
#include "pc/audio_track.h"
#include "pc/jitter_buffer_delay.h"
#include "pc/media_stream_track_proxy.h"
#include "pc/remote_audio_source.h"
#include "pc/rtp_receiver.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class AudioRtpReceiver : public ObserverInterface,
                         public AudioSourceInterface::AudioObserver,
                         public RtpReceiverInternal {
 public:
  // The constructor supports optionally passing the voice channel to the
  // instance at construction time without having to call `SetMediaChannel()`
  // on the worker thread straight after construction.
  // However, when using that, the assumption is that right after construction,
  // a call to either `SetupUnsignaledMediaChannel` or `SetupMediaChannel`
  // will be made, which will internally start the source on the worker thread.
  AudioRtpReceiver(
      rtc::Thread* worker_thread,
      std::string receiver_id,
      std::vector<std::string> stream_ids,
      bool is_unified_plan,
      cricket::VoiceMediaReceiveChannelInterface* voice_channel = nullptr);
  // TODO(https://crbug.com/webrtc/9480): Remove this when streams() is removed.
  AudioRtpReceiver(
      rtc::Thread* worker_thread,
      const std::string& receiver_id,
      const std::vector<rtc::scoped_refptr<MediaStreamInterface>>& streams,
      bool is_unified_plan,
      cricket::VoiceMediaReceiveChannelInterface* media_channel = nullptr);
  virtual ~AudioRtpReceiver();

  // ObserverInterface implementation
  void OnChanged() override;

  // AudioSourceInterface::AudioObserver implementation
  void OnSetVolume(double volume) override;

  rtc::scoped_refptr<AudioTrackInterface> audio_track() const { return track_; }

  // RtpReceiverInterface implementation
  rtc::scoped_refptr<MediaStreamTrackInterface> track() const override {
    return track_;
  }
  rtc::scoped_refptr<DtlsTransportInterface> dtls_transport() const override;
  std::vector<std::string> stream_ids() const override;
  std::vector<rtc::scoped_refptr<MediaStreamInterface>> streams()
      const override;

  cricket::MediaType media_type() const override {
    return cricket::MEDIA_TYPE_AUDIO;
  }

  std::string id() const override { return id_; }

  RtpParameters GetParameters() const override;

  void SetFrameDecryptor(
      rtc::scoped_refptr<FrameDecryptorInterface> frame_decryptor) override;

  rtc::scoped_refptr<FrameDecryptorInterface> GetFrameDecryptor()
      const override;

  // RtpReceiverInternal implementation.
  void Stop() override;
  void SetupMediaChannel(uint32_t ssrc) override;
  void SetupUnsignaledMediaChannel() override;
  std::optional<uint32_t> ssrc() const override;
  void NotifyFirstPacketReceived() override;
  void set_stream_ids(std::vector<std::string> stream_ids) override;
  void set_transport(
      rtc::scoped_refptr<DtlsTransportInterface> dtls_transport) override;
  void SetStreams(const std::vector<rtc::scoped_refptr<MediaStreamInterface>>&
                      streams) override;
  void SetObserver(RtpReceiverObserverInterface* observer) override;

  void SetJitterBufferMinimumDelay(
      std::optional<double> delay_seconds) override;

  void SetMediaChannel(
      cricket::MediaReceiveChannelInterface* media_channel) override;

  std::vector<RtpSource> GetSources() const override;
  int AttachmentId() const override { return attachment_id_; }
  void SetFrameTransformer(
      rtc::scoped_refptr<FrameTransformerInterface> frame_transformer) override;

 private:
  void RestartMediaChannel(std::optional<uint32_t> ssrc)
      RTC_RUN_ON(&signaling_thread_checker_);
  void RestartMediaChannel_w(std::optional<uint32_t> ssrc,
                             bool track_enabled,
                             MediaSourceInterface::SourceState state)
      RTC_RUN_ON(worker_thread_);
  void Reconfigure(bool track_enabled) RTC_RUN_ON(worker_thread_);
  void SetOutputVolume_w(double volume) RTC_RUN_ON(worker_thread_);

  RTC_NO_UNIQUE_ADDRESS SequenceChecker signaling_thread_checker_;
  rtc::Thread* const worker_thread_;
  const std::string id_;
  const rtc::scoped_refptr<RemoteAudioSource> source_;
  const rtc::scoped_refptr<AudioTrackProxyWithInternal<AudioTrack>> track_;
  cricket::VoiceMediaReceiveChannelInterface* media_channel_
      RTC_GUARDED_BY(worker_thread_) = nullptr;
  std::optional<uint32_t> signaled_ssrc_ RTC_GUARDED_BY(worker_thread_);
  std::vector<rtc::scoped_refptr<MediaStreamInterface>> streams_
      RTC_GUARDED_BY(&signaling_thread_checker_);
  bool cached_track_enabled_ RTC_GUARDED_BY(&signaling_thread_checker_);
  double cached_volume_ RTC_GUARDED_BY(worker_thread_) = 1.0;
  RtpReceiverObserverInterface* observer_
      RTC_GUARDED_BY(&signaling_thread_checker_) = nullptr;
  bool received_first_packet_ RTC_GUARDED_BY(&signaling_thread_checker_) =
      false;
  const int attachment_id_;
  rtc::scoped_refptr<FrameDecryptorInterface> frame_decryptor_
      RTC_GUARDED_BY(worker_thread_);
  rtc::scoped_refptr<DtlsTransportInterface> dtls_transport_
      RTC_GUARDED_BY(&signaling_thread_checker_);
  // Stores and updates the playout delay. Handles caching cases if
  // `SetJitterBufferMinimumDelay` is called before start.
  JitterBufferDelay delay_ RTC_GUARDED_BY(worker_thread_);
  rtc::scoped_refptr<FrameTransformerInterface> frame_transformer_
      RTC_GUARDED_BY(worker_thread_);
  const rtc::scoped_refptr<PendingTaskSafetyFlag> worker_thread_safety_;
};

}  // namespace webrtc

#endif  // PC_AUDIO_RTP_RECEIVER_H_
