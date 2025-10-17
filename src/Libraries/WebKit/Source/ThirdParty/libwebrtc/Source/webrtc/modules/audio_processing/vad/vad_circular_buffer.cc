/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include "modules/audio_processing/vad/vad_circular_buffer.h"

#include <stdlib.h>

namespace webrtc {

VadCircularBuffer::VadCircularBuffer(int buffer_size)
    : buffer_(new double[buffer_size]),
      is_full_(false),
      index_(0),
      buffer_size_(buffer_size),
      sum_(0) {}

VadCircularBuffer::~VadCircularBuffer() {}

void VadCircularBuffer::Reset() {
  is_full_ = false;
  index_ = 0;
  sum_ = 0;
}

VadCircularBuffer* VadCircularBuffer::Create(int buffer_size) {
  if (buffer_size <= 0)
    return NULL;
  return new VadCircularBuffer(buffer_size);
}

double VadCircularBuffer::Oldest() const {
  if (!is_full_)
    return buffer_[0];
  else
    return buffer_[index_];
}

double VadCircularBuffer::Mean() {
  double m;
  if (is_full_) {
    m = sum_ / buffer_size_;
  } else {
    if (index_ > 0)
      m = sum_ / index_;
    else
      m = 0;
  }
  return m;
}

void VadCircularBuffer::Insert(double value) {
  if (is_full_) {
    sum_ -= buffer_[index_];
  }
  sum_ += value;
  buffer_[index_] = value;
  index_++;
  if (index_ >= buffer_size_) {
    is_full_ = true;
    index_ = 0;
  }
}
int VadCircularBuffer::BufferLevel() {
  if (is_full_)
    return buffer_size_;
  return index_;
}

int VadCircularBuffer::Get(int index, double* value) const {
  int err = ConvertToLinearIndex(&index);
  if (err < 0)
    return -1;
  *value = buffer_[index];
  return 0;
}

int VadCircularBuffer::Set(int index, double value) {
  int err = ConvertToLinearIndex(&index);
  if (err < 0)
    return -1;

  sum_ -= buffer_[index];
  buffer_[index] = value;
  sum_ += value;
  return 0;
}

int VadCircularBuffer::ConvertToLinearIndex(int* index) const {
  if (*index < 0 || *index >= buffer_size_)
    return -1;

  if (!is_full_ && *index >= index_)
    return -1;

  *index = index_ - 1 - *index;
  if (*index < 0)
    *index += buffer_size_;
  return 0;
}

int VadCircularBuffer::RemoveTransient(int width_threshold,
                                       double val_threshold) {
  if (!is_full_ && index_ < width_threshold + 2)
    return 0;

  int index_1 = 0;
  int index_2 = width_threshold + 1;
  double v = 0;
  if (Get(index_1, &v) < 0)
    return -1;
  if (v < val_threshold) {
    Set(index_1, 0);
    int index;
    for (index = index_2; index > index_1; index--) {
      if (Get(index, &v) < 0)
        return -1;
      if (v < val_threshold)
        break;
    }
    for (; index > index_1; index--) {
      if (Set(index, 0.0) < 0)
        return -1;
    }
  }
  return 0;
}

}  // namespace webrtc
