/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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
#pragma once

#include <unistd.h>

#include "platform/bionic/macros.h"
#include "private/ErrnoRestorer.h"

class ScopedFd final {
 public:
  explicit ScopedFd(int fd) : fd_(fd) {
  }

  ScopedFd() : fd_(-1) {
  }

  ~ScopedFd() {
    reset(-1);
  }

  void reset(int fd = -1) {
    if (fd_ != -1) {
      ErrnoRestorer e;
      close(fd_);
    }
    fd_ = fd;
  }

  int get() const {
    return fd_;
  }

 private:
  int fd_;

  BIONIC_DISALLOW_COPY_AND_ASSIGN(ScopedFd);
};
