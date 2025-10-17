/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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

#include <android/fdsan.h>

#include <errno.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/cdefs.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/user.h>

struct FdEntry {
  _Atomic(uint64_t) close_tag = 0;
};

struct FdTableOverflow {
  size_t len = 0;
  FdEntry entries[0];
};

template <size_t inline_fds>
struct FdTableImpl {
  constexpr FdTableImpl() {}

  uint32_t version = 0;  // currently 0, and hopefully it'll stay that way.
  _Atomic(android_fdsan_error_level) error_level = ANDROID_FDSAN_ERROR_LEVEL_DISABLED;

  FdEntry entries[inline_fds];
  _Atomic(FdTableOverflow*) overflow = nullptr;

  FdEntry* at(size_t idx);
};

using FdTable = FdTableImpl<128>;
