/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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

class PropertiesFilename {
 public:
  PropertiesFilename() = default;
  PropertiesFilename(const char* dir, const char* file) {
    if (snprintf(filename_, sizeof(filename_), "%s/%s", dir, file) >=
        static_cast<int>(sizeof(filename_))) {
      abort();
    }
  }
  void operator=(const char* value) {
    if (strlen(value) >= sizeof(filename_)) abort();
    strcpy(filename_, value);
  }
  const char* c_str() { return filename_; }

 private:
  // Typically something like "/dev/__properties__/properties_serial", but can be as long as
  // "/data/local/tmp/TemporaryDir-fntJb8/appcompat_override/u:object_r:PROPERTY_NAME_prop:s0"
  // when running CTS.
  char filename_[256];
};
