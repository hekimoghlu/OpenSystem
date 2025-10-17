/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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

// Android's 32-bit ABI shipped with a sigset_t too small to include any
// of the realtime signals, so we have both sigset_t and sigset64_t. Many
// new system calls only accept a sigset64_t, so this helps paper over
// the difference at zero cost to LP64 in most cases after the optimizer
// removes the unnecessary temporary `ptr`.
struct SigSetConverter {
 public:
  SigSetConverter(const sigset_t* s) : SigSetConverter(const_cast<sigset_t*>(s)) {}

  SigSetConverter(sigset_t* s) {
#if defined(__LP64__)
    // sigset_t == sigset64_t on LP64.
    ptr = s;
#else
    sigset64 = {};
    if (s != nullptr) {
      original_ptr = s;
      sigset = *s;
      ptr = &sigset64;
    } else {
      ptr = nullptr;
    }
#endif
  }

  void copy_out() {
#if defined(__LP64__)
    // We used the original pointer directly, so no copy needed.
#else
    *original_ptr = sigset;
#endif
  }

  sigset64_t* ptr;

 private:
  [[maybe_unused]] sigset_t* original_ptr;
  union {
    sigset_t sigset;
    sigset64_t sigset64;
  };
};
