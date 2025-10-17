/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

#include "private/bionic_lock.h"

#include "prop_area.h"

class ContextNode {
 public:
  ContextNode(const char* context, const char* filename)
      : context_(context), pa_(nullptr), no_access_(false), filename_(filename) {
    lock_.init(false);
  }
  ~ContextNode() {
    Unmap();
  }

  BIONIC_DISALLOW_COPY_AND_ASSIGN(ContextNode);

  bool Open(bool access_rw, bool* fsetxattr_failed);
  bool CheckAccessAndOpen();
  void ResetAccess();
  void Unmap();

  const char* context() const {
    return context_;
  }
  prop_area* pa() {
    return pa_;
  }

 private:
  bool CheckAccess();

  Lock lock_;
  const char* context_;
  prop_area* pa_;
  bool no_access_;
  const char* filename_;
};
