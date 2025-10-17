/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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

#include <stdint.h>
#include <string.h>

#include <vector>

#include <platform/bionic/macros.h>

#include "OptionData.h"

// Forward declarations.
class DebugData;
struct Header;
class Config;

class GuardData : public OptionData {
 public:
  GuardData(DebugData* debug_data, int init_value, size_t num_bytes);
  virtual ~GuardData() = default;

  bool Valid(void* data) { return memcmp(data, cmp_mem_.data(), cmp_mem_.size()) == 0; }

  void LogFailure(const Header* header, const void* pointer, const void* data);

 protected:
  std::vector<uint8_t> cmp_mem_;

  virtual const char* GetTypeName() = 0;

  BIONIC_DISALLOW_COPY_AND_ASSIGN(GuardData);
};

class FrontGuardData : public GuardData {
 public:
  FrontGuardData(DebugData* debug_data, const Config& config, size_t* offset);
  virtual ~FrontGuardData() = default;

  bool Valid(const Header* header);

  void LogFailure(const Header* header);

  size_t offset() { return offset_; }

 private:
  const char* GetTypeName() override { return "FRONT"; }

  size_t offset_ = 0;

  BIONIC_DISALLOW_COPY_AND_ASSIGN(FrontGuardData);
};

class RearGuardData : public GuardData {
 public:
  RearGuardData(DebugData* debug_data, const Config& config);
  virtual ~RearGuardData() = default;

  bool Valid(const Header* header);

  void LogFailure(const Header* header);

 private:
  const char* GetTypeName() override { return "REAR"; }

  BIONIC_DISALLOW_COPY_AND_ASSIGN(RearGuardData);
};
