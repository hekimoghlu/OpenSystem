/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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
#include "rtc_base/copy_on_write_buffer.h"

#include <stddef.h>

#include "absl/strings/string_view.h"

namespace rtc {

CopyOnWriteBuffer::CopyOnWriteBuffer() : offset_(0), size_(0) {
  RTC_DCHECK(IsConsistent());
}

CopyOnWriteBuffer::CopyOnWriteBuffer(const CopyOnWriteBuffer& buf)
    : buffer_(buf.buffer_), offset_(buf.offset_), size_(buf.size_) {}

CopyOnWriteBuffer::CopyOnWriteBuffer(CopyOnWriteBuffer&& buf) noexcept
    : buffer_(std::move(buf.buffer_)), offset_(buf.offset_), size_(buf.size_) {
  buf.offset_ = 0;
  buf.size_ = 0;
  RTC_DCHECK(IsConsistent());
}

CopyOnWriteBuffer::CopyOnWriteBuffer(absl::string_view s)
    : CopyOnWriteBuffer(s.data(), s.length()) {}

CopyOnWriteBuffer::CopyOnWriteBuffer(size_t size)
    : buffer_(size > 0 ? new RefCountedBuffer(size) : nullptr),
      offset_(0),
      size_(size) {
  RTC_DCHECK(IsConsistent());
}

CopyOnWriteBuffer::CopyOnWriteBuffer(size_t size, size_t capacity)
    : buffer_(size > 0 || capacity > 0 ? new RefCountedBuffer(size, capacity)
                                       : nullptr),
      offset_(0),
      size_(size) {
  RTC_DCHECK(IsConsistent());
}

CopyOnWriteBuffer::~CopyOnWriteBuffer() = default;

bool CopyOnWriteBuffer::operator==(const CopyOnWriteBuffer& buf) const {
  // Must either be the same view of the same buffer or have the same contents.
  RTC_DCHECK(IsConsistent());
  RTC_DCHECK(buf.IsConsistent());
  return size_ == buf.size_ &&
         (cdata() == buf.cdata() || memcmp(cdata(), buf.cdata(), size_) == 0);
}

void CopyOnWriteBuffer::SetSize(size_t size) {
  RTC_DCHECK(IsConsistent());
  if (!buffer_) {
    if (size > 0) {
      buffer_ = new RefCountedBuffer(size);
      offset_ = 0;
      size_ = size;
    }
    RTC_DCHECK(IsConsistent());
    return;
  }

  if (size <= size_) {
    size_ = size;
    return;
  }

  UnshareAndEnsureCapacity(std::max(capacity(), size));
  buffer_->SetSize(size + offset_);
  size_ = size;
  RTC_DCHECK(IsConsistent());
}

void CopyOnWriteBuffer::EnsureCapacity(size_t new_capacity) {
  RTC_DCHECK(IsConsistent());
  if (!buffer_) {
    if (new_capacity > 0) {
      buffer_ = new RefCountedBuffer(0, new_capacity);
      offset_ = 0;
      size_ = 0;
    }
    RTC_DCHECK(IsConsistent());
    return;
  } else if (new_capacity <= capacity()) {
    return;
  }

  UnshareAndEnsureCapacity(new_capacity);
  RTC_DCHECK(IsConsistent());
}

void CopyOnWriteBuffer::Clear() {
  if (!buffer_)
    return;

  if (buffer_->HasOneRef()) {
    buffer_->Clear();
  } else {
    buffer_ = new RefCountedBuffer(0, capacity());
  }
  offset_ = 0;
  size_ = 0;
  RTC_DCHECK(IsConsistent());
}

void CopyOnWriteBuffer::UnshareAndEnsureCapacity(size_t new_capacity) {
  if (buffer_->HasOneRef() && new_capacity <= capacity()) {
    return;
  }

  buffer_ =
      new RefCountedBuffer(buffer_->data() + offset_, size_, new_capacity);
  offset_ = 0;
  RTC_DCHECK(IsConsistent());
}

}  // namespace rtc
