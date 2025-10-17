/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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

#ifndef MKVMUXER_MKVWRITER_H_
#define MKVMUXER_MKVWRITER_H_

#include <stdio.h>

#include "mkvmuxer/mkvmuxer.h"
#include "mkvmuxer/mkvmuxertypes.h"

namespace mkvmuxer {

// Default implementation of the IMkvWriter interface on Windows.
class MkvWriter : public IMkvWriter {
 public:
  MkvWriter();
  explicit MkvWriter(FILE* fp);
  virtual ~MkvWriter();

  // IMkvWriter interface
  virtual int64 Position() const;
  virtual int32 Position(int64 position);
  virtual bool Seekable() const;
  virtual int32 Write(const void* buffer, uint32 length);
  virtual void ElementStartNotify(uint64 element_id, int64 position);

  // Creates and opens a file for writing. |filename| is the name of the file
  // to open. This function will overwrite the contents of |filename|. Returns
  // true on success.
  bool Open(const char* filename);

  // Closes an opened file.
  void Close();

 private:
  // File handle to output file.
  FILE* file_;
  bool writer_owns_file_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(MkvWriter);
};

}  // namespace mkvmuxer

#endif  // MKVMUXER_MKVWRITER_H_
