/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

#include "extraDataFields.h"
#include <language/Runtime/Metadata.h>

using namespace language;

namespace toolchain {
  int DisableABIBreakingChecks;
  int EnableABIBreakingChecks;
}

void *completeMetadata(void *uncastMetadata) {
  auto *metadata = static_cast<Metadata *>(uncastMetadata);

  auto request =
        MetadataRequest(MetadataState::Complete, /*isNonBlocking*/ false);
  auto response = language_checkMetadataState(request, metadata);
  return (void *)response.Value;
}

int64_t *trailingFlagsForStructMetadata(void *uncastMetadata) {
  auto *metadata = static_cast<StructMetadata *>(uncastMetadata);
  auto *description = metadata->getDescription();
  if (!description->isGeneric())
    return nullptr;

  auto *trailingFlags = metadata->getTrailingFlags();
  if (trailingFlags == nullptr)
    return nullptr;

  auto *retval = (int64_t *)malloc(sizeof(int64_t));
  *retval = trailingFlags->getOpaqueValue();
  return retval;
}

const uint32_t *fieldOffsetsForStructMetadata(void *uncastMetadata) {
  auto *metadata = static_cast<StructMetadata *>(uncastMetadata);
  return metadata->getFieldOffsets();
}

int64_t *trailingFlagsForEnumMetadata(void *uncastMetadata) {
  auto *metadata = static_cast<EnumMetadata *>(uncastMetadata);
  auto *description = metadata->getDescription();
  if (!description->isGeneric())
    return nullptr;

  auto *trailingFlags = metadata->getTrailingFlags();
  if (trailingFlags == nullptr)
    return nullptr;

  auto *retval = (int64_t *)malloc(sizeof(int64_t));
  *retval = trailingFlags->getOpaqueValue();
  return retval;
}

size_t *payloadSizeForEnumMetadata(void *uncastMetadata) {
  auto *metadata = static_cast<EnumMetadata *>(uncastMetadata);
  if (!metadata->hasPayloadSize())
    return nullptr;

  auto *retval = (size_t *)malloc(sizeof(size_t));
  *retval = metadata->getPayloadSize();
  return retval;
}

