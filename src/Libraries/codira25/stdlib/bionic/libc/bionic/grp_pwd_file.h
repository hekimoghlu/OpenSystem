/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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

#include <grp.h>
#include <pwd.h>

#include "private/bionic_lock.h"
#include "platform/bionic/macros.h"
#include "private/grp_pwd.h"

class MmapFile {
 public:
  MmapFile(const char* filename, const char* required_prefix);

  template <typename Line>
  bool FindById(uid_t uid, Line* line);
  template <typename Line>
  bool FindByName(const char* name, Line* line);
  void Unmap();

  BIONIC_DISALLOW_IMPLICIT_CONSTRUCTORS(MmapFile);

 private:
  enum class FileStatus {
    Uninitialized,
    Initialized,
    Error,
  };

  bool GetFile(const char** start, const char** end);
  bool DoMmap();

  template <typename Line, typename Predicate>
  bool Find(Line* line, Predicate predicate);

  FileStatus status_ = FileStatus::Uninitialized;
  Lock lock_;
  const char* filename_ = nullptr;
  const char* start_ = nullptr;
  const char* end_ = nullptr;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
  const char* required_prefix_;
#pragma clang diagnostic pop
};

class PasswdFile {
 public:
  PasswdFile(const char* filename, const char* required_prefix);

  bool FindById(uid_t id, passwd_state_t* passwd_state);
  bool FindByName(const char* name, passwd_state_t* passwd_state);
  void Unmap() {
    mmap_file_.Unmap();
  }

  BIONIC_DISALLOW_IMPLICIT_CONSTRUCTORS(PasswdFile);

 private:
  MmapFile mmap_file_;
};

class GroupFile {
 public:
  GroupFile(const char* filename, const char* required_prefix);

  bool FindById(gid_t id, group_state_t* group_state);
  bool FindByName(const char* name, group_state_t* group_state);
  void Unmap() {
    mmap_file_.Unmap();
  }

  BIONIC_DISALLOW_IMPLICIT_CONSTRUCTORS(GroupFile);

 private:
  MmapFile mmap_file_;
};
