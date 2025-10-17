/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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

#include <dirent.h>

#include "platform/bionic/macros.h"

class ScopedReaddir {
 public:
  explicit ScopedReaddir(const char* path) : ScopedReaddir(opendir(path)) {
  }

  explicit ScopedReaddir(DIR* dir) {
    dir_ = dir;
  }

  ~ScopedReaddir() {
    if (dir_ != nullptr) {
      closedir(dir_);
    }
  }

  bool IsBad() {
    return dir_ == nullptr;
  }

  dirent* ReadEntry() {
    return readdir(dir_);
  }

 private:
  DIR* dir_;

  BIONIC_DISALLOW_COPY_AND_ASSIGN(ScopedReaddir);
};
