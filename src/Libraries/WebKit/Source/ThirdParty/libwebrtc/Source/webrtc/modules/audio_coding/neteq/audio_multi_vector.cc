/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#include "modules/audio_coding/neteq/audio_multi_vector.h"

#include <algorithm>

#include "rtc_base/checks.h"

namespace webrtc {

AudioMultiVector::AudioMultiVector(size_t N) {
  RTC_DCHECK_GT(N, 0);
  if (N < 1)
    N = 1;
  for (size_t n = 0; n < N; ++n) {
    channels_.push_back(new AudioVector);
  }
  num_channels_ = N;
}

AudioMultiVector::AudioMultiVector(size_t N, size_t initial_size) {
  RTC_DCHECK_GT(N, 0);
  if (N < 1)
    N = 1;
  for (size_t n = 0; n < N; ++n) {
    channels_.push_back(new AudioVector(initial_size));
  }
  num_channels_ = N;
}

AudioMultiVector::~AudioMultiVector() {
  std::vector<AudioVector*>::iterator it = channels_.begin();
  while (it != channels_.end()) {
    delete (*it);
    ++it;
  }
}

void AudioMultiVector::Clear() {
  for (size_t i = 0; i < num_channels_; ++i) {
    channels_[i]->Clear();
  }
}

void AudioMultiVector::Zeros(size_t length) {
  for (size_t i = 0; i < num_channels_; ++i) {
    channels_[i]->Clear();
    channels_[i]->Extend(length);
  }
}

void AudioMultiVector::CopyTo(AudioMultiVector* copy_to) const {
  if (copy_to) {
    for (size_t i = 0; i < num_channels_; ++i) {
      channels_[i]->CopyTo(&(*copy_to)[i]);
    }
  }
}

void AudioMultiVector::PushBackInterleaved(
    rtc::ArrayView<const int16_t> append_this) {
  RTC_DCHECK_EQ(append_this.size() % num_channels_, 0);
  if (append_this.empty()) {
    return;
  }
  if (num_channels_ == 1) {
    // Special case to avoid extra allocation and data shuffling.
    channels_[0]->PushBack(append_this.data(), append_this.size());
    return;
  }
  size_t length_per_channel = append_this.size() / num_channels_;
  int16_t* temp_array = new int16_t[length_per_channel];  // Temporary storage.
  for (size_t channel = 0; channel < num_channels_; ++channel) {
    // Copy elements to `temp_array`.
    for (size_t i = 0; i < length_per_channel; ++i) {
      temp_array[i] = append_this[channel + i * num_channels_];
    }
    channels_[channel]->PushBack(temp_array, length_per_channel);
  }
  delete[] temp_array;
}

void AudioMultiVector::PushBack(const AudioMultiVector& append_this) {
  RTC_DCHECK_EQ(num_channels_, append_this.num_channels_);
  if (num_channels_ == append_this.num_channels_) {
    for (size_t i = 0; i < num_channels_; ++i) {
      channels_[i]->PushBack(append_this[i]);
    }
  }
}

void AudioMultiVector::PushBackFromIndex(const AudioMultiVector& append_this,
                                         size_t index) {
  RTC_DCHECK_LT(index, append_this.Size());
  index = std::min(index, append_this.Size() - 1);
  size_t length = append_this.Size() - index;
  RTC_DCHECK_EQ(num_channels_, append_this.num_channels_);
  if (num_channels_ == append_this.num_channels_) {
    for (size_t i = 0; i < num_channels_; ++i) {
      channels_[i]->PushBack(append_this[i], length, index);
    }
  }
}

void AudioMultiVector::PopFront(size_t length) {
  for (size_t i = 0; i < num_channels_; ++i) {
    channels_[i]->PopFront(length);
  }
}

void AudioMultiVector::PopBack(size_t length) {
  for (size_t i = 0; i < num_channels_; ++i) {
    channels_[i]->PopBack(length);
  }
}

size_t AudioMultiVector::ReadInterleaved(size_t length,
                                         int16_t* destination) const {
  return ReadInterleavedFromIndex(0, length, destination);
}

size_t AudioMultiVector::ReadInterleavedFromIndex(size_t start_index,
                                                  size_t length,
                                                  int16_t* destination) const {
  RTC_DCHECK(destination);
  size_t index = 0;  // Number of elements written to `destination` so far.
  RTC_DCHECK_LE(start_index, Size());
  start_index = std::min(start_index, Size());
  if (length + start_index > Size()) {
    length = Size() - start_index;
  }
  if (num_channels_ == 1) {
    // Special case to avoid the nested for loop below.
    (*this)[0].CopyTo(length, start_index, destination);
    return length;
  }
  for (size_t i = 0; i < length; ++i) {
    for (size_t channel = 0; channel < num_channels_; ++channel) {
      destination[index] = (*this)[channel][i + start_index];
      ++index;
    }
  }
  return index;
}

size_t AudioMultiVector::ReadInterleavedFromEnd(size_t length,
                                                int16_t* destination) const {
  length = std::min(length, Size());  // Cannot read more than Size() elements.
  return ReadInterleavedFromIndex(Size() - length, length, destination);
}

void AudioMultiVector::OverwriteAt(const AudioMultiVector& insert_this,
                                   size_t length,
                                   size_t position) {
  RTC_DCHECK_EQ(num_channels_, insert_this.num_channels_);
  // Cap `length` at the length of `insert_this`.
  RTC_DCHECK_LE(length, insert_this.Size());
  length = std::min(length, insert_this.Size());
  if (num_channels_ == insert_this.num_channels_) {
    for (size_t i = 0; i < num_channels_; ++i) {
      channels_[i]->OverwriteAt(insert_this[i], length, position);
    }
  }
}

void AudioMultiVector::CrossFade(const AudioMultiVector& append_this,
                                 size_t fade_length) {
  RTC_DCHECK_EQ(num_channels_, append_this.num_channels_);
  if (num_channels_ == append_this.num_channels_) {
    for (size_t i = 0; i < num_channels_; ++i) {
      channels_[i]->CrossFade(append_this[i], fade_length);
    }
  }
}

size_t AudioMultiVector::Channels() const {
  return num_channels_;
}

size_t AudioMultiVector::Size() const {
  RTC_DCHECK(channels_[0]);
  return channels_[0]->Size();
}

void AudioMultiVector::AssertSize(size_t required_size) {
  if (Size() < required_size) {
    size_t extend_length = required_size - Size();
    for (size_t channel = 0; channel < num_channels_; ++channel) {
      channels_[channel]->Extend(extend_length);
    }
  }
}

bool AudioMultiVector::Empty() const {
  RTC_DCHECK(channels_[0]);
  return channels_[0]->Empty();
}

void AudioMultiVector::CopyChannel(size_t from_channel, size_t to_channel) {
  RTC_DCHECK_LT(from_channel, num_channels_);
  RTC_DCHECK_LT(to_channel, num_channels_);
  channels_[from_channel]->CopyTo(channels_[to_channel]);
}

const AudioVector& AudioMultiVector::operator[](size_t index) const {
  return *(channels_[index]);
}

AudioVector& AudioMultiVector::operator[](size_t index) {
  return *(channels_[index]);
}

}  // namespace webrtc
