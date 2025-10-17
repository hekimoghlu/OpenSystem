/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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

#include "isPrespecialized.h"
#include <language/Runtime/Metadata.h>

using namespace language;

namespace toolchain {
  int DisableABIBreakingChecks;
  int EnableABIBreakingChecks;
}

bool isCanonicalStaticallySpecializedGenericMetadata(Metadata *self) {
  if (auto *metadata = dyn_cast<StructMetadata>(self)) {
    return metadata->isCanonicalStaticallySpecializedGenericMetadata();
  }
  if (auto *metadata = dyn_cast<EnumMetadata>(self))
    return metadata->isCanonicalStaticallySpecializedGenericMetadata();
  if (auto *metadata = dyn_cast<ClassMetadata>(self))
    return metadata->isCanonicalStaticallySpecializedGenericMetadata();

  return false;
}

bool isCanonicalStaticallySpecializedGenericMetadata(void *ptr) {
  auto metadata = static_cast<Metadata *>(ptr);
  auto isCanonical =
      isCanonicalStaticallySpecializedGenericMetadata(metadata);
  return isCanonical;
}

bool isStaticallySpecializedGenericMetadata(Metadata *self) {
  if (auto *metadata = dyn_cast<StructMetadata>(self)) {
    return metadata->isStaticallySpecializedGenericMetadata();
  }
  if (auto *metadata = dyn_cast<EnumMetadata>(self))
    return metadata->isStaticallySpecializedGenericMetadata();
  if (auto *metadata = dyn_cast<ClassMetadata>(self))
    return metadata->isStaticallySpecializedGenericMetadata();

  return false;
}

bool isStaticallySpecializedGenericMetadata(void *ptr) {
  auto metadata = static_cast<Metadata *>(ptr);
  auto isStatic =
      isStaticallySpecializedGenericMetadata(metadata);
  return isStatic;
}

void allocateDirtyAndFreeChunk(void) {
  uintptr_t ChunkSize = 16 * 1024;

  char *allocation = new char[ChunkSize];

  memset(allocation, 255, ChunkSize);

  delete[] allocation;
}
