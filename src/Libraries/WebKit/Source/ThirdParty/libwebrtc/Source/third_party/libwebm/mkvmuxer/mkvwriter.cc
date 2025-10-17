/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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

// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include "mkvmuxer/mkvwriter.h"

#include <sys/types.h>

#ifdef _MSC_VER
#include <share.h>  // for _SH_DENYWR
#endif

namespace mkvmuxer {

MkvWriter::MkvWriter() : file_(NULL), writer_owns_file_(true) {}

MkvWriter::MkvWriter(FILE* fp) : file_(fp), writer_owns_file_(false) {}

MkvWriter::~MkvWriter() { Close(); }

int32 MkvWriter::Write(const void* buffer, uint32 length) {
  if (!file_)
    return -1;

  if (length == 0)
    return 0;

  if (buffer == NULL)
    return -1;

  const size_t bytes_written = fwrite(buffer, 1, length, file_);

  return (bytes_written == length) ? 0 : -1;
}

bool MkvWriter::Open(const char* filename) {
  if (filename == NULL)
    return false;

  if (file_)
    return false;

#ifdef _MSC_VER
  file_ = _fsopen(filename, "wb", _SH_DENYWR);
#else
  file_ = fopen(filename, "wb");
#endif
  if (file_ == NULL)
    return false;
  return true;
}

void MkvWriter::Close() {
  if (file_ && writer_owns_file_) {
    fclose(file_);
  }
  file_ = NULL;
}

int64 MkvWriter::Position() const {
  if (!file_)
    return 0;

#ifdef _MSC_VER
  return _ftelli64(file_);
#else
  return ftell(file_);
#endif
}

int32 MkvWriter::Position(int64 position) {
  if (!file_)
    return -1;

#ifdef _MSC_VER
  return _fseeki64(file_, position, SEEK_SET);
#elif defined(_WIN32)
  return fseeko64(file_, static_cast<off_t>(position), SEEK_SET);
#else
  return fseeko(file_, static_cast<off_t>(position), SEEK_SET);
#endif
}

bool MkvWriter::Seekable() const { return true; }

void MkvWriter::ElementStartNotify(uint64, int64) {}

}  // namespace mkvmuxer
