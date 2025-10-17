/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
#include "system_properties/context_node.h"

#include <limits.h>
#include <unistd.h>

#include <async_safe/log.h>

#include "system_properties/system_properties.h"

// pthread_mutex_lock() calls into system_properties in the case of contention.
// This creates a risk of dead lock if any system_properties functions
// use pthread locks after system_property initialization.
//
// For this reason, the below three functions use a bionic Lock and static
// allocation of memory for each filename.

bool ContextNode::Open(bool access_rw, bool* fsetxattr_failed) {
  lock_.lock();
  if (pa_) {
    lock_.unlock();
    return true;
  }

  PropertiesFilename filename(filename_, context_);
  if (access_rw) {
    pa_ = prop_area::map_prop_area_rw(filename.c_str(), context_, fsetxattr_failed);
  } else {
    pa_ = prop_area::map_prop_area(filename.c_str());
  }
  lock_.unlock();
  return pa_;
}

bool ContextNode::CheckAccessAndOpen() {
  if (!pa_ && !no_access_) {
    if (!CheckAccess() || !Open(false, nullptr)) {
      no_access_ = true;
    }
  }
  return pa_;
}

void ContextNode::ResetAccess() {
  if (!CheckAccess()) {
    Unmap();
    no_access_ = true;
  } else {
    no_access_ = false;
  }
}

bool ContextNode::CheckAccess() {
  PropertiesFilename filename(filename_, context_);
  return access(filename.c_str(), R_OK) == 0;
}

void ContextNode::Unmap() {
  prop_area::unmap_prop_area(&pa_);
}
