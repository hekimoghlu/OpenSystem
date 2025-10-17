/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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

#include "language/Core/Basic/Cuda.h"

#include "toolchain/ADT/Twine.h"
#include "toolchain/Support/ErrorHandling.h"
#include "toolchain/Support/VersionTuple.h"

namespace language::Core {

struct CudaVersionMapEntry {
  const char *Name;
  CudaVersion Version;
  toolchain::VersionTuple TVersion;
};
#define CUDA_ENTRY(major, minor)                                               \
  {                                                                            \
    #major "." #minor, CudaVersion::CUDA_##major##minor,                       \
        toolchain::VersionTuple(major, minor)                                       \
  }

static const CudaVersionMapEntry CudaNameVersionMap[] = {
    CUDA_ENTRY(7, 0),
    CUDA_ENTRY(7, 5),
    CUDA_ENTRY(8, 0),
    CUDA_ENTRY(9, 0),
    CUDA_ENTRY(9, 1),
    CUDA_ENTRY(9, 2),
    CUDA_ENTRY(10, 0),
    CUDA_ENTRY(10, 1),
    CUDA_ENTRY(10, 2),
    CUDA_ENTRY(11, 0),
    CUDA_ENTRY(11, 1),
    CUDA_ENTRY(11, 2),
    CUDA_ENTRY(11, 3),
    CUDA_ENTRY(11, 4),
    CUDA_ENTRY(11, 5),
    CUDA_ENTRY(11, 6),
    CUDA_ENTRY(11, 7),
    CUDA_ENTRY(11, 8),
    CUDA_ENTRY(12, 0),
    CUDA_ENTRY(12, 1),
    CUDA_ENTRY(12, 2),
    CUDA_ENTRY(12, 3),
    CUDA_ENTRY(12, 4),
    CUDA_ENTRY(12, 5),
    CUDA_ENTRY(12, 6),
    CUDA_ENTRY(12, 8),
    CUDA_ENTRY(12, 9),
    {"", CudaVersion::NEW, toolchain::VersionTuple(std::numeric_limits<int>::max())},
    {"unknown", CudaVersion::UNKNOWN, {}} // End of list tombstone.
};
#undef CUDA_ENTRY

const char *CudaVersionToString(CudaVersion V) {
  for (auto *I = CudaNameVersionMap; I->Version != CudaVersion::UNKNOWN; ++I)
    if (I->Version == V)
      return I->Name;

  return CudaVersionToString(CudaVersion::UNKNOWN);
}

CudaVersion CudaStringToVersion(const toolchain::Twine &S) {
  std::string VS = S.str();
  for (auto *I = CudaNameVersionMap; I->Version != CudaVersion::UNKNOWN; ++I)
    if (I->Name == VS)
      return I->Version;
  return CudaVersion::UNKNOWN;
}

CudaVersion ToCudaVersion(toolchain::VersionTuple Version) {
  for (auto *I = CudaNameVersionMap; I->Version != CudaVersion::UNKNOWN; ++I)
    if (I->TVersion == Version)
      return I->Version;
  return CudaVersion::UNKNOWN;
}

CudaVersion MinVersionForOffloadArch(OffloadArch A) {
  if (A == OffloadArch::UNKNOWN)
    return CudaVersion::UNKNOWN;

  // AMD GPUs do not depend on CUDA versions.
  if (IsAMDOffloadArch(A))
    return CudaVersion::CUDA_70;

  switch (A) {
  case OffloadArch::SM_20:
  case OffloadArch::SM_21:
  case OffloadArch::SM_30:
  case OffloadArch::SM_32_:
  case OffloadArch::SM_35:
  case OffloadArch::SM_37:
  case OffloadArch::SM_50:
  case OffloadArch::SM_52:
  case OffloadArch::SM_53:
    return CudaVersion::CUDA_70;
  case OffloadArch::SM_60:
  case OffloadArch::SM_61:
  case OffloadArch::SM_62:
    return CudaVersion::CUDA_80;
  case OffloadArch::SM_70:
    return CudaVersion::CUDA_90;
  case OffloadArch::SM_72:
    return CudaVersion::CUDA_91;
  case OffloadArch::SM_75:
    return CudaVersion::CUDA_100;
  case OffloadArch::SM_80:
    return CudaVersion::CUDA_110;
  case OffloadArch::SM_86:
    return CudaVersion::CUDA_111;
  case OffloadArch::SM_87:
    return CudaVersion::CUDA_114;
  case OffloadArch::SM_89:
  case OffloadArch::SM_90:
    return CudaVersion::CUDA_118;
  case OffloadArch::SM_90a:
    return CudaVersion::CUDA_120;
  case OffloadArch::SM_100:
  case OffloadArch::SM_100a:
  case OffloadArch::SM_101:
  case OffloadArch::SM_101a:
  case OffloadArch::SM_120:
  case OffloadArch::SM_120a:
    return CudaVersion::CUDA_128;
  case OffloadArch::SM_103:
  case OffloadArch::SM_103a:
  case OffloadArch::SM_121:
  case OffloadArch::SM_121a:
    return CudaVersion::CUDA_129;
  default:
    toolchain_unreachable("invalid enum");
  }
}

CudaVersion MaxVersionForOffloadArch(OffloadArch A) {
  // AMD GPUs do not depend on CUDA versions.
  if (IsAMDOffloadArch(A))
    return CudaVersion::NEW;

  switch (A) {
  case OffloadArch::UNKNOWN:
    return CudaVersion::UNKNOWN;
  case OffloadArch::SM_20:
  case OffloadArch::SM_21:
    return CudaVersion::CUDA_80;
  case OffloadArch::SM_30:
  case OffloadArch::SM_32_:
    return CudaVersion::CUDA_102;
  case OffloadArch::SM_35:
  case OffloadArch::SM_37:
    return CudaVersion::CUDA_118;
  default:
    return CudaVersion::NEW;
  }
}

bool CudaFeatureEnabled(toolchain::VersionTuple Version, CudaFeature Feature) {
  return CudaFeatureEnabled(ToCudaVersion(Version), Feature);
}

bool CudaFeatureEnabled(CudaVersion Version, CudaFeature Feature) {
  switch (Feature) {
  case CudaFeature::CUDA_USES_NEW_LAUNCH:
    return Version >= CudaVersion::CUDA_92;
  case CudaFeature::CUDA_USES_FATBIN_REGISTER_END:
    return Version >= CudaVersion::CUDA_101;
  }
  toolchain_unreachable("Unknown CUDA feature.");
}
} // namespace language::Core
