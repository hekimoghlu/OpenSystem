/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

//===--- NVPTX.cpp - Implement NVPTX target feature support ---------------===//
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
// This file implements NVPTX TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "language/Core/Basic/Builtins.h"
#include "language/Core/Basic/MacroBuilder.h"
#include "language/Core/Basic/TargetBuiltins.h"
#include "toolchain/ADT/StringSwitch.h"

using namespace language::Core;
using namespace language::Core::targets;

static constexpr int NumBuiltins =
    language::Core::NVPTX::LastTSBuiltin - Builtin::FirstTSBuiltin;

#define GET_BUILTIN_STR_TABLE
#include "language/Core/Basic/BuiltinsNVPTX.inc"
#undef GET_BUILTIN_STR_TABLE

static constexpr Builtin::Info BuiltinInfos[] = {
#define GET_BUILTIN_INFOS
#include "language/Core/Basic/BuiltinsNVPTX.inc"
#undef GET_BUILTIN_INFOS
};
static_assert(std::size(BuiltinInfos) == NumBuiltins);

const char *const NVPTXTargetInfo::GCCRegNames[] = {"r0"};

NVPTXTargetInfo::NVPTXTargetInfo(const toolchain::Triple &Triple,
                                 const TargetOptions &Opts,
                                 unsigned TargetPointerWidth)
    : TargetInfo(Triple) {
  assert((TargetPointerWidth == 32 || TargetPointerWidth == 64) &&
         "NVPTX only supports 32- and 64-bit modes.");

  PTXVersion = 32;
  for (const StringRef Feature : Opts.FeaturesAsWritten) {
    int PTXV;
    if (!Feature.starts_with("+ptx") ||
        Feature.drop_front(4).getAsInteger(10, PTXV))
      continue;
    PTXVersion = PTXV; // TODO: should it be max(PTXVersion, PTXV)?
  }

  TLSSupported = false;
  VLASupported = false;
  AddrSpaceMap = &NVPTXAddrSpaceMap;
  UseAddrSpaceMapMangling = true;
  // __bf16 is always available as a load/store only type.
  BFloat16Width = BFloat16Align = 16;
  BFloat16Format = &toolchain::APFloat::BFloat();

  // Define available target features
  // These must be defined in sorted order!
  NoAsmVariants = true;
  GPU = OffloadArch::UNUSED;

  // PTX supports f16 as a fundamental type.
  HasLegalHalfType = true;
  HasFloat16 = true;

  if (TargetPointerWidth == 32)
    resetDataLayout(
        "e-p:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");
  else if (Opts.NVPTXUseShortPointers)
    resetDataLayout(
        "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:"
        "16-v32:32-n16:32:64");
  else
    resetDataLayout("e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");

  // If possible, get a TargetInfo for our host triple, so we can match its
  // types.
  toolchain::Triple HostTriple(Opts.HostTriple);
  if (!HostTriple.isNVPTX())
    HostTarget = AllocateTarget(toolchain::Triple(Opts.HostTriple), Opts);

  // If no host target, make some guesses about the data layout and return.
  if (!HostTarget) {
    LongWidth = LongAlign = TargetPointerWidth;
    PointerWidth = PointerAlign = TargetPointerWidth;
    switch (TargetPointerWidth) {
    case 32:
      SizeType = TargetInfo::UnsignedInt;
      PtrDiffType = TargetInfo::SignedInt;
      IntPtrType = TargetInfo::SignedInt;
      break;
    case 64:
      SizeType = TargetInfo::UnsignedLong;
      PtrDiffType = TargetInfo::SignedLong;
      IntPtrType = TargetInfo::SignedLong;
      break;
    default:
      toolchain_unreachable("TargetPointerWidth must be 32 or 64");
    }

    MaxAtomicInlineWidth = TargetPointerWidth;
    return;
  }

  // Copy properties from host target.
  PointerWidth = HostTarget->getPointerWidth(LangAS::Default);
  PointerAlign = HostTarget->getPointerAlign(LangAS::Default);
  BoolWidth = HostTarget->getBoolWidth();
  BoolAlign = HostTarget->getBoolAlign();
  IntWidth = HostTarget->getIntWidth();
  IntAlign = HostTarget->getIntAlign();
  HalfWidth = HostTarget->getHalfWidth();
  HalfAlign = HostTarget->getHalfAlign();
  FloatWidth = HostTarget->getFloatWidth();
  FloatAlign = HostTarget->getFloatAlign();
  DoubleWidth = HostTarget->getDoubleWidth();
  DoubleAlign = HostTarget->getDoubleAlign();
  LongWidth = HostTarget->getLongWidth();
  LongAlign = HostTarget->getLongAlign();
  LongLongWidth = HostTarget->getLongLongWidth();
  LongLongAlign = HostTarget->getLongLongAlign();
  MinGlobalAlign = HostTarget->getMinGlobalAlign(/* TypeSize = */ 0,
                                                 /* HasNonWeakDef = */ true);
  NewAlign = HostTarget->getNewAlign();
  DefaultAlignForAttributeAligned =
      HostTarget->getDefaultAlignForAttributeAligned();
  SizeType = HostTarget->getSizeType();
  IntMaxType = HostTarget->getIntMaxType();
  PtrDiffType = HostTarget->getPtrDiffType(LangAS::Default);
  IntPtrType = HostTarget->getIntPtrType();
  WCharType = HostTarget->getWCharType();
  WIntType = HostTarget->getWIntType();
  Char16Type = HostTarget->getChar16Type();
  Char32Type = HostTarget->getChar32Type();
  Int64Type = HostTarget->getInt64Type();
  SigAtomicType = HostTarget->getSigAtomicType();
  ProcessIDType = HostTarget->getProcessIDType();

  UseBitFieldTypeAlignment = HostTarget->useBitFieldTypeAlignment();
  UseZeroLengthBitfieldAlignment = HostTarget->useZeroLengthBitfieldAlignment();
  UseExplicitBitFieldAlignment = HostTarget->useExplicitBitFieldAlignment();
  ZeroLengthBitfieldBoundary = HostTarget->getZeroLengthBitfieldBoundary();

  // This is a bit of a lie, but it controls __GCC_ATOMIC_XXX_LOCK_FREE, and
  // we need those macros to be identical on host and device, because (among
  // other things) they affect which standard library classes are defined, and
  // we need all classes to be defined on both the host and device.
  MaxAtomicInlineWidth = HostTarget->getMaxAtomicInlineWidth();

  // Properties intentionally not copied from host:
  // - LargeArrayMinWidth, LargeArrayAlign: Not visible across the
  //   host/device boundary.
  // - SuitableAlign: Not visible across the host/device boundary, and may
  //   correctly be different on host/device, e.g. if host has wider vector
  //   types than device.
  // - LongDoubleWidth, LongDoubleAlign: nvptx's long double type is the same
  //   as its double type, but that's not necessarily true on the host.
  //   TODO: nvcc emits a warning when using long double on device; we should
  //   do the same.
}

ArrayRef<const char *> NVPTXTargetInfo::getGCCRegNames() const {
  return toolchain::ArrayRef(GCCRegNames);
}

bool NVPTXTargetInfo::hasFeature(StringRef Feature) const {
  return toolchain::StringSwitch<bool>(Feature)
      .Cases("ptx", "nvptx", true)
      .Default(false);
}

void NVPTXTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__PTX__");
  Builder.defineMacro("__NVPTX__");

  // Skip setting architecture dependent macros if undefined.
  if (GPU == OffloadArch::UNUSED && !HostTarget)
    return;

  if (Opts.CUDAIsDevice || Opts.OpenMPIsTargetDevice || !HostTarget) {
    // Set __CUDA_ARCH__ for the GPU specified.
    toolchain::StringRef CUDAArchCode = [this] {
      switch (GPU) {
      case OffloadArch::GFX600:
      case OffloadArch::GFX601:
      case OffloadArch::GFX602:
      case OffloadArch::GFX700:
      case OffloadArch::GFX701:
      case OffloadArch::GFX702:
      case OffloadArch::GFX703:
      case OffloadArch::GFX704:
      case OffloadArch::GFX705:
      case OffloadArch::GFX801:
      case OffloadArch::GFX802:
      case OffloadArch::GFX803:
      case OffloadArch::GFX805:
      case OffloadArch::GFX810:
      case OffloadArch::GFX9_GENERIC:
      case OffloadArch::GFX900:
      case OffloadArch::GFX902:
      case OffloadArch::GFX904:
      case OffloadArch::GFX906:
      case OffloadArch::GFX908:
      case OffloadArch::GFX909:
      case OffloadArch::GFX90a:
      case OffloadArch::GFX90c:
      case OffloadArch::GFX9_4_GENERIC:
      case OffloadArch::GFX942:
      case OffloadArch::GFX950:
      case OffloadArch::GFX10_1_GENERIC:
      case OffloadArch::GFX1010:
      case OffloadArch::GFX1011:
      case OffloadArch::GFX1012:
      case OffloadArch::GFX1013:
      case OffloadArch::GFX10_3_GENERIC:
      case OffloadArch::GFX1030:
      case OffloadArch::GFX1031:
      case OffloadArch::GFX1032:
      case OffloadArch::GFX1033:
      case OffloadArch::GFX1034:
      case OffloadArch::GFX1035:
      case OffloadArch::GFX1036:
      case OffloadArch::GFX11_GENERIC:
      case OffloadArch::GFX1100:
      case OffloadArch::GFX1101:
      case OffloadArch::GFX1102:
      case OffloadArch::GFX1103:
      case OffloadArch::GFX1150:
      case OffloadArch::GFX1151:
      case OffloadArch::GFX1152:
      case OffloadArch::GFX1153:
      case OffloadArch::GFX12_GENERIC:
      case OffloadArch::GFX1200:
      case OffloadArch::GFX1201:
      case OffloadArch::GFX1250:
      case OffloadArch::AMDGCNSPIRV:
      case OffloadArch::Generic:
      case OffloadArch::GRANITERAPIDS:
      case OffloadArch::BMG_G21:
      case OffloadArch::LAST:
        break;
      case OffloadArch::UNKNOWN:
        assert(false && "No GPU arch when compiling CUDA device code.");
        return "";
      case OffloadArch::UNUSED:
      case OffloadArch::SM_20:
        return "200";
      case OffloadArch::SM_21:
        return "210";
      case OffloadArch::SM_30:
        return "300";
      case OffloadArch::SM_32_:
        return "320";
      case OffloadArch::SM_35:
        return "350";
      case OffloadArch::SM_37:
        return "370";
      case OffloadArch::SM_50:
        return "500";
      case OffloadArch::SM_52:
        return "520";
      case OffloadArch::SM_53:
        return "530";
      case OffloadArch::SM_60:
        return "600";
      case OffloadArch::SM_61:
        return "610";
      case OffloadArch::SM_62:
        return "620";
      case OffloadArch::SM_70:
        return "700";
      case OffloadArch::SM_72:
        return "720";
      case OffloadArch::SM_75:
        return "750";
      case OffloadArch::SM_80:
        return "800";
      case OffloadArch::SM_86:
        return "860";
      case OffloadArch::SM_87:
        return "870";
      case OffloadArch::SM_89:
        return "890";
      case OffloadArch::SM_90:
      case OffloadArch::SM_90a:
        return "900";
      case OffloadArch::SM_100:
      case OffloadArch::SM_100a:
        return "1000";
      case OffloadArch::SM_101:
      case OffloadArch::SM_101a:
        return "1010";
      case OffloadArch::SM_103:
      case OffloadArch::SM_103a:
        return "1030";
      case OffloadArch::SM_120:
      case OffloadArch::SM_120a:
        return "1200";
      case OffloadArch::SM_121:
      case OffloadArch::SM_121a:
        return "1210";
      }
      toolchain_unreachable("unhandled OffloadArch");
    }();
    Builder.defineMacro("__CUDA_ARCH__", CUDAArchCode);
    switch(GPU) {
      case OffloadArch::SM_90a:
      case OffloadArch::SM_100a:
      case OffloadArch::SM_101a:
      case OffloadArch::SM_103a:
      case OffloadArch::SM_120a:
      case OffloadArch::SM_121a:
        Builder.defineMacro("__CUDA_ARCH_FEAT_SM" + CUDAArchCode.drop_back() + "_ALL", "1");
        break;
      default:
        // Do nothing if this is not an enhanced architecture.
        break;
    }
  }
}

toolchain::SmallVector<Builtin::InfosShard>
NVPTXTargetInfo::getTargetBuiltins() const {
  return {{&BuiltinStrings, BuiltinInfos}};
}
