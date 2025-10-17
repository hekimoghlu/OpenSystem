/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#include "modules/audio_device/fine_audio_buffer.h"

#include <cstdint>
#include <cstring>

#include "api/array_view.h"
#include "modules/audio_device/audio_device_buffer.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

FineAudioBuffer::FineAudioBuffer(AudioDeviceBuffer* audio_device_buffer)
    : audio_device_buffer_(audio_device_buffer),
      playout_samples_per_channel_10ms_(rtc::dchecked_cast<size_t>(
          audio_device_buffer->PlayoutSampleRate() * 10 / 1000)),
      record_samples_per_channel_10ms_(rtc::dchecked_cast<size_t>(
          audio_device_buffer->RecordingSampleRate() * 10 / 1000)),
      playout_channels_(audio_device_buffer->PlayoutChannels()),
      record_channels_(audio_device_buffer->RecordingChannels()) {
  RTC_DCHECK(audio_device_buffer_);
  RTC_DLOG(LS_INFO) << __FUNCTION__;
  if (IsReadyForPlayout()) {
    RTC_DLOG(LS_INFO) << "playout_samples_per_channel_10ms: "
                      << playout_samples_per_channel_10ms_;
    RTC_DLOG(LS_INFO) << "playout_channels: " << playout_channels_;
  }
  if (IsReadyForRecord()) {
    RTC_DLOG(LS_INFO) << "record_samples_per_channel_10ms: "
                      << record_samples_per_channel_10ms_;
    RTC_DLOG(LS_INFO) << "record_channels: " << record_channels_;
  }
}

FineAudioBuffer::~FineAudioBuffer() {
  RTC_DLOG(LS_INFO) << __FUNCTION__;
}

void FineAudioBuffer::ResetPlayout() {
  playout_buffer_.Clear();
}

void FineAudioBuffer::ResetRecord() {
  record_buffer_.Clear();
}

bool FineAudioBuffer::IsReadyForPlayout() const {
  return playout_samples_per_channel_10ms_ > 0 && playout_channels_ > 0;
}

bool FineAudioBuffer::IsReadyForRecord() const {
  return record_samples_per_channel_10ms_ > 0 && record_channels_ > 0;
}

void FineAudioBuffer::GetPlayoutData(rtc::ArrayView<int16_t> audio_buffer,
                                     int playout_delay_ms) {
  RTC_DCHECK(IsReadyForPlayout());
  // Ask WebRTC for new data in chunks of 10ms until we have enough to
  // fulfill the request. It is possible that the buffer already contains
  // enough samples from the last round.
  while (playout_buffer_.size() < audio_buffer.size()) {
    // Get 10ms decoded audio from WebRTC. The ADB knows about number of
    // channels; hence we can ask for number of samples per channel here.
    if (audio_device_buffer_->RequestPlayoutData(
            playout_samples_per_channel_10ms_) ==
        static_cast<int32_t>(playout_samples_per_channel_10ms_)) {
      // Append 10ms to the end of the local buffer taking number of channels
      // into account.
      const size_t num_elements_10ms =
          playout_channels_ * playout_samples_per_channel_10ms_;
      const size_t written_elements = playout_buffer_.AppendData(
          num_elements_10ms, [&](rtc::ArrayView<int16_t> buf) {
            const size_t samples_per_channel_10ms =
                audio_device_buffer_->GetPlayoutData(buf.data());
            return playout_channels_ * samples_per_channel_10ms;
          });
      RTC_DCHECK_EQ(num_elements_10ms, written_elements);
    } else {
      // Provide silence if AudioDeviceBuffer::RequestPlayoutData() fails.
      // Can e.g. happen when an AudioTransport has not been registered.
      const size_t num_bytes = audio_buffer.size() * sizeof(int16_t);
      std::memset(audio_buffer.data(), 0, num_bytes);
      return;
    }
  }

  // Provide the requested number of bytes to the consumer.
  const size_t num_bytes = audio_buffer.size() * sizeof(int16_t);
  memcpy(audio_buffer.data(), playout_buffer_.data(), num_bytes);
  // Move remaining samples to start of buffer to prepare for next round.
  memmove(playout_buffer_.data(), playout_buffer_.data() + audio_buffer.size(),
          (playout_buffer_.size() - audio_buffer.size()) * sizeof(int16_t));
  playout_buffer_.SetSize(playout_buffer_.size() - audio_buffer.size());
  // Cache playout latency for usage in DeliverRecordedData();
  playout_delay_ms_ = playout_delay_ms;
}

void FineAudioBuffer::DeliverRecordedData(
    rtc::ArrayView<const int16_t> audio_buffer,
    int record_delay_ms,
    std::optional<int64_t> capture_time_ns) {
  RTC_DCHECK(IsReadyForRecord());
  // Always append new data and grow the buffer when needed.
  record_buffer_.AppendData(audio_buffer.data(), audio_buffer.size());
  // Consume samples from buffer in chunks of 10ms until there is not
  // enough data left. The number of remaining samples in the cache is given by
  // the new size of the internal `record_buffer_`.
  const size_t num_elements_10ms =
      record_channels_ * record_samples_per_channel_10ms_;
  while (record_buffer_.size() >= num_elements_10ms) {
    audio_device_buffer_->SetRecordedBuffer(record_buffer_.data(),
                                            record_samples_per_channel_10ms_,
                                            capture_time_ns);
    audio_device_buffer_->SetVQEData(playout_delay_ms_, record_delay_ms);
    audio_device_buffer_->DeliverRecordedData();
    memmove(record_buffer_.data(), record_buffer_.data() + num_elements_10ms,
            (record_buffer_.size() - num_elements_10ms) * sizeof(int16_t));
    record_buffer_.SetSize(record_buffer_.size() - num_elements_10ms);
  }
}

}  // namespace webrtc
