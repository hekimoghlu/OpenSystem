/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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

//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#include <IndexStoreDB_LLVMSupport/toolchain_Support_ARMBuildAttributes.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_TargetParser.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_ArrayRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringSwitch.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_Twine.h>

using namespace toolchain;
using namespace AMDGPU;

namespace {

struct GPUInfo {
  StringLiteral Name;
  StringLiteral CanonicalName;
  AMDGPU::GPUKind Kind;
  unsigned Features;
};

constexpr GPUInfo R600GPUs[26] = {
  // Name       Canonical    Kind        Features
  //            Name
  {{"r600"},    {"r600"},    GK_R600,    FEATURE_NONE },
  {{"rv630"},   {"r600"},    GK_R600,    FEATURE_NONE },
  {{"rv635"},   {"r600"},    GK_R600,    FEATURE_NONE },
  {{"r630"},    {"r630"},    GK_R630,    FEATURE_NONE },
  {{"rs780"},   {"rs880"},   GK_RS880,   FEATURE_NONE },
  {{"rs880"},   {"rs880"},   GK_RS880,   FEATURE_NONE },
  {{"rv610"},   {"rs880"},   GK_RS880,   FEATURE_NONE },
  {{"rv620"},   {"rs880"},   GK_RS880,   FEATURE_NONE },
  {{"rv670"},   {"rv670"},   GK_RV670,   FEATURE_NONE },
  {{"rv710"},   {"rv710"},   GK_RV710,   FEATURE_NONE },
  {{"rv730"},   {"rv730"},   GK_RV730,   FEATURE_NONE },
  {{"rv740"},   {"rv770"},   GK_RV770,   FEATURE_NONE },
  {{"rv770"},   {"rv770"},   GK_RV770,   FEATURE_NONE },
  {{"cedar"},   {"cedar"},   GK_CEDAR,   FEATURE_NONE },
  {{"palm"},    {"cedar"},   GK_CEDAR,   FEATURE_NONE },
  {{"cypress"}, {"cypress"}, GK_CYPRESS, FEATURE_FMA  },
  {{"hemlock"}, {"cypress"}, GK_CYPRESS, FEATURE_FMA  },
  {{"juniper"}, {"juniper"}, GK_JUNIPER, FEATURE_NONE },
  {{"redwood"}, {"redwood"}, GK_REDWOOD, FEATURE_NONE },
  {{"sumo"},    {"sumo"},    GK_SUMO,    FEATURE_NONE },
  {{"sumo2"},   {"sumo"},    GK_SUMO,    FEATURE_NONE },
  {{"barts"},   {"barts"},   GK_BARTS,   FEATURE_NONE },
  {{"caicos"},  {"caicos"},  GK_CAICOS,  FEATURE_NONE },
  {{"aruba"},   {"cayman"},  GK_CAYMAN,  FEATURE_FMA  },
  {{"cayman"},  {"cayman"},  GK_CAYMAN,  FEATURE_FMA  },
  {{"turks"},   {"turks"},   GK_TURKS,   FEATURE_NONE }
};

// This table should be sorted by the value of GPUKind
// Don't bother listing the implicitly true features
constexpr GPUInfo AMDGCNGPUs[33] = {
  // Name         Canonical    Kind        Features
  //              Name
  {{"gfx600"},    {"gfx600"},  GK_GFX600,  FEATURE_FAST_FMA_F32},
  {{"tahiti"},    {"gfx600"},  GK_GFX600,  FEATURE_FAST_FMA_F32},
  {{"gfx601"},    {"gfx601"},  GK_GFX601,  FEATURE_NONE},
  {{"hainan"},    {"gfx601"},  GK_GFX601,  FEATURE_NONE},
  {{"oland"},     {"gfx601"},  GK_GFX601,  FEATURE_NONE},
  {{"pitcairn"},  {"gfx601"},  GK_GFX601,  FEATURE_NONE},
  {{"verde"},     {"gfx601"},  GK_GFX601,  FEATURE_NONE},
  {{"gfx700"},    {"gfx700"},  GK_GFX700,  FEATURE_NONE},
  {{"kaveri"},    {"gfx700"},  GK_GFX700,  FEATURE_NONE},
  {{"gfx701"},    {"gfx701"},  GK_GFX701,  FEATURE_FAST_FMA_F32},
  {{"hawaii"},    {"gfx701"},  GK_GFX701,  FEATURE_FAST_FMA_F32},
  {{"gfx702"},    {"gfx702"},  GK_GFX702,  FEATURE_FAST_FMA_F32},
  {{"gfx703"},    {"gfx703"},  GK_GFX703,  FEATURE_NONE},
  {{"kabini"},    {"gfx703"},  GK_GFX703,  FEATURE_NONE},
  {{"mullins"},   {"gfx703"},  GK_GFX703,  FEATURE_NONE},
  {{"gfx704"},    {"gfx704"},  GK_GFX704,  FEATURE_NONE},
  {{"bonaire"},   {"gfx704"},  GK_GFX704,  FEATURE_NONE},
  {{"gfx801"},    {"gfx801"},  GK_GFX801,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32},
  {{"carrizo"},   {"gfx801"},  GK_GFX801,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32},
  {{"gfx802"},    {"gfx802"},  GK_GFX802,  FEATURE_FAST_DENORMAL_F32},
  {{"iceland"},   {"gfx802"},  GK_GFX802,  FEATURE_FAST_DENORMAL_F32},
  {{"tonga"},     {"gfx802"},  GK_GFX802,  FEATURE_FAST_DENORMAL_F32},
  {{"gfx803"},    {"gfx803"},  GK_GFX803,  FEATURE_FAST_DENORMAL_F32},
  {{"fiji"},      {"gfx803"},  GK_GFX803,  FEATURE_FAST_DENORMAL_F32},
  {{"polaris10"}, {"gfx803"},  GK_GFX803,  FEATURE_FAST_DENORMAL_F32},
  {{"polaris11"}, {"gfx803"},  GK_GFX803,  FEATURE_FAST_DENORMAL_F32},
  {{"gfx810"},    {"gfx810"},  GK_GFX810,  FEATURE_FAST_DENORMAL_F32},
  {{"stoney"},    {"gfx810"},  GK_GFX810,  FEATURE_FAST_DENORMAL_F32},
  {{"gfx900"},    {"gfx900"},  GK_GFX900,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32},
  {{"gfx902"},    {"gfx902"},  GK_GFX902,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32},
  {{"gfx904"},    {"gfx904"},  GK_GFX904,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32},
  {{"gfx906"},    {"gfx906"},  GK_GFX906,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32},
  {{"gfx909"},    {"gfx909"},  GK_GFX909,  FEATURE_FAST_FMA_F32|FEATURE_FAST_DENORMAL_F32},
};

const GPUInfo *getArchEntry(AMDGPU::GPUKind AK, ArrayRef<GPUInfo> Table) {
  GPUInfo Search = { {""}, {""}, AK, AMDGPU::FEATURE_NONE };

  auto I = std::lower_bound(Table.begin(), Table.end(), Search,
    [](const GPUInfo &A, const GPUInfo &B) {
      return A.Kind < B.Kind;
    });

  if (I == Table.end())
    return nullptr;
  return I;
}

} // namespace

StringRef toolchain::AMDGPU::getArchNameAMDGCN(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, AMDGCNGPUs))
    return Entry->CanonicalName;
  return "";
}

StringRef toolchain::AMDGPU::getArchNameR600(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, R600GPUs))
    return Entry->CanonicalName;
  return "";
}

AMDGPU::GPUKind toolchain::AMDGPU::parseArchAMDGCN(StringRef CPU) {
  for (const auto C : AMDGCNGPUs) {
    if (CPU == C.Name)
      return C.Kind;
  }

  return AMDGPU::GPUKind::GK_NONE;
}

AMDGPU::GPUKind toolchain::AMDGPU::parseArchR600(StringRef CPU) {
  for (const auto C : R600GPUs) {
    if (CPU == C.Name)
      return C.Kind;
  }

  return AMDGPU::GPUKind::GK_NONE;
}

unsigned AMDGPU::getArchAttrAMDGCN(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, AMDGCNGPUs))
    return Entry->Features;
  return FEATURE_NONE;
}

unsigned AMDGPU::getArchAttrR600(GPUKind AK) {
  if (const auto *Entry = getArchEntry(AK, R600GPUs))
    return Entry->Features;
  return FEATURE_NONE;
}

void AMDGPU::fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &Values) {
  // XXX: Should this only report unique canonical names?
  for (const auto C : AMDGCNGPUs)
    Values.push_back(C.Name);
}

void AMDGPU::fillValidArchListR600(SmallVectorImpl<StringRef> &Values) {
  for (const auto C : R600GPUs)
    Values.push_back(C.Name);
}

AMDGPU::IsaVersion AMDGPU::getIsaVersion(StringRef GPU) {
  AMDGPU::GPUKind AK = parseArchAMDGCN(GPU);
  if (AK == AMDGPU::GPUKind::GK_NONE) {
    if (GPU == "generic-hsa")
      return {7, 0, 0};
    if (GPU == "generic")
      return {6, 0, 0};
    return {0, 0, 0};
  }

  switch (AK) {
  case GK_GFX600: return {6, 0, 0};
  case GK_GFX601: return {6, 0, 1};
  case GK_GFX700: return {7, 0, 0};
  case GK_GFX701: return {7, 0, 1};
  case GK_GFX702: return {7, 0, 2};
  case GK_GFX703: return {7, 0, 3};
  case GK_GFX704: return {7, 0, 4};
  case GK_GFX801: return {8, 0, 1};
  case GK_GFX802: return {8, 0, 2};
  case GK_GFX803: return {8, 0, 3};
  case GK_GFX810: return {8, 1, 0};
  case GK_GFX900: return {9, 0, 0};
  case GK_GFX902: return {9, 0, 2};
  case GK_GFX904: return {9, 0, 4};
  case GK_GFX906: return {9, 0, 6};
  case GK_GFX909: return {9, 0, 9};
  default:        return {0, 0, 0};
  }
}
