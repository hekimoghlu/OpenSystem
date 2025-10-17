/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

#ifndef WEBVTT_VTTREADER_H_
#define WEBVTT_VTTREADER_H_

#include <cstdio>
#include "./webvttparser.h"

namespace libwebvtt {

class VttReader : public libwebvtt::Reader {
 public:
  VttReader();
  virtual ~VttReader();

  // Open the file identified by |filename| in read-only mode, as a
  // binary stream of bytes.  Returns 0 on success, negative if error.
  int Open(const char* filename);

  // Closes the file stream.  Note that the stream is automatically
  // closed when the VttReader object is destroyed.
  void Close();

  // Reads the next character in the file stream, as per the semantics
  // of Reader::GetChar.  Returns negative if error, 0 on success, and
  // positive if end-of-stream has been reached.
  virtual int GetChar(char* c);

 private:
  FILE* file_;

  VttReader(const VttReader&);
  VttReader& operator=(const VttReader&);
};

}  // namespace libwebvtt

#endif  // WEBVTT_VTTREADER_H_
