/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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

#include <sys/param.h>
#include <sys/system_properties.h>

#include "contexts.h"
#include "contexts_pre_split.h"
#include "contexts_serialized.h"
#include "contexts_split.h"

class SystemProperties {
 public:
  friend struct LocalPropertyTestState;
  friend class SystemPropertiesTest;
  // Note that system properties are initialized before libc calls static initializers, so
  // doing any initialization in this constructor is an error.  Even a Constructor that zero
  // initializes this class will clobber the previous property initialization.
  // We rely on the static SystemProperties in libc to be placed in .bss and zero initialized.
  SystemProperties() = default;
  // Special constructor for testing that also zero initializes the important members.
  explicit SystemProperties(bool initialized) : initialized_(initialized) {
  }

  BIONIC_DISALLOW_COPY_AND_ASSIGN(SystemProperties);

  bool Init(const char* filename);
  bool Reload(bool load_default_path);
  bool AreaInit(const char* filename, bool* fsetxattr_failed);
  bool AreaInit(const char* filename, bool* fsetxattr_failed, bool load_default_path);
  uint32_t AreaSerial();
  const prop_info* Find(const char* name);
  int Read(const prop_info* pi, char* name, char* value);
  void ReadCallback(const prop_info* pi,
                    void (*callback)(void* cookie, const char* name, const char* value,
                                     uint32_t serial),
                    void* cookie);
  int Get(const char* name, char* value);
  int Update(prop_info* pi, const char* value, unsigned int len);
  int Add(const char* name, unsigned int namelen, const char* value, unsigned int valuelen);
  uint32_t WaitAny(uint32_t old_serial);
  bool Wait(const prop_info* pi, uint32_t old_serial, uint32_t* new_serial_ptr,
            const timespec* relative_timeout);
  const prop_info* FindNth(unsigned n);
  int Foreach(void (*propfn)(const prop_info* pi, void* cookie), void* cookie);

 private:
  uint32_t ReadMutablePropertyValue(const prop_info* pi, char* value);

  // We don't want to use new or malloc in properties (b/31659220), and we don't want to waste a
  // full page by using mmap(), so we set aside enough space to create any context of the three
  // contexts.
  static constexpr size_t kMaxContextsAlign =
      MAX(alignof(ContextsSerialized), MAX(alignof(ContextsSplit), alignof(ContextsPreSplit)));
  static constexpr size_t kMaxContextsSize =
      MAX(sizeof(ContextsSerialized), MAX(sizeof(ContextsSplit), sizeof(ContextsPreSplit)));
  alignas(kMaxContextsAlign) char contexts_data_[kMaxContextsSize];
  alignas(kMaxContextsAlign) char appcompat_override_contexts_data_[kMaxContextsSize];
  Contexts* contexts_;
  // See http://b/291816546#comment#3 for more explanation of appcompat_override
  Contexts* appcompat_override_contexts_;

  bool InitContexts(bool load_default_path);

  bool initialized_;
  PropertiesFilename properties_filename_;
  PropertiesFilename appcompat_filename_;
};
