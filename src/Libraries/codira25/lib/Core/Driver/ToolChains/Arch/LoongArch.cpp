/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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

//===--- LoongArch.cpp - LoongArch Helpers for Tools ------------*- C++ -*-===//
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

#include "LoongArch.h"
#include "../Clang.h"
#include "language/Core/Basic/DiagnosticDriver.h"
#include "language/Core/Driver/CommonArgs.h"
#include "language/Core/Driver/Driver.h"
#include "language/Core/Driver/Options.h"
#include "toolchain/TargetParser/Host.h"
#include "toolchain/TargetParser/LoongArchTargetParser.h"

using namespace language::Core::driver;
using namespace language::Core::driver::tools;
using namespace language::Core;
using namespace toolchain::opt;

StringRef loongarch::getLoongArchABI(const Driver &D, const ArgList &Args,
                                     const toolchain::Triple &Triple) {
  assert((Triple.getArch() == toolchain::Triple::loongarch32 ||
          Triple.getArch() == toolchain::Triple::loongarch64) &&
         "Unexpected triple");
  bool IsLA32 = Triple.getArch() == toolchain::Triple::loongarch32;

  // Record -mabi value for later use.
  const Arg *MABIArg = Args.getLastArg(options::OPT_mabi_EQ);
  StringRef MABIValue;
  if (MABIArg) {
    MABIValue = MABIArg->getValue();
  }

  // Parse -mfpu value for later use.
  const Arg *MFPUArg = Args.getLastArg(options::OPT_mfpu_EQ);
  int FPU = -1;
  if (MFPUArg) {
    StringRef V = MFPUArg->getValue();
    if (V == "64")
      FPU = 64;
    else if (V == "32")
      FPU = 32;
    else if (V == "0" || V == "none")
      FPU = 0;
    else
      D.Diag(diag::err_drv_loongarch_invalid_mfpu_EQ) << V;
  }

  // Check -m*-float firstly since they have highest priority.
  if (const Arg *A = Args.getLastArg(options::OPT_mdouble_float,
                                     options::OPT_msingle_float,
                                     options::OPT_msoft_float)) {
    StringRef ImpliedABI;
    int ImpliedFPU = -1;
    if (A->getOption().matches(options::OPT_mdouble_float)) {
      ImpliedABI = IsLA32 ? "ilp32d" : "lp64d";
      ImpliedFPU = 64;
    }
    if (A->getOption().matches(options::OPT_msingle_float)) {
      ImpliedABI = IsLA32 ? "ilp32f" : "lp64f";
      ImpliedFPU = 32;
    }
    if (A->getOption().matches(options::OPT_msoft_float)) {
      ImpliedABI = IsLA32 ? "ilp32s" : "lp64s";
      ImpliedFPU = 0;
    }

    // Check `-mabi=` and `-mfpu=` settings and report if they conflict with
    // the higher-priority settings implied by -m*-float.
    //
    // ImpliedABI and ImpliedFPU are guaranteed to have valid values because
    // one of the match arms must match if execution can arrive here at all.
    if (!MABIValue.empty() && ImpliedABI != MABIValue)
      D.Diag(diag::warn_drv_loongarch_conflicting_implied_val)
          << MABIArg->getAsString(Args) << A->getAsString(Args) << ImpliedABI;

    if (FPU != -1 && ImpliedFPU != FPU)
      D.Diag(diag::warn_drv_loongarch_conflicting_implied_val)
          << MFPUArg->getAsString(Args) << A->getAsString(Args) << ImpliedFPU;

    return ImpliedABI;
  }

  // If `-mabi=` is specified, use it.
  if (!MABIValue.empty())
    return MABIValue;

  // Select abi based on -mfpu=xx.
  switch (FPU) {
  case 64:
    return IsLA32 ? "ilp32d" : "lp64d";
  case 32:
    return IsLA32 ? "ilp32f" : "lp64f";
  case 0:
    return IsLA32 ? "ilp32s" : "lp64s";
  }

  // Choose a default based on the triple.
  // Honor the explicit ABI modifier suffix in triple's environment part if
  // present, falling back to {ILP32,LP64}D otherwise.
  switch (Triple.getEnvironment()) {
  case toolchain::Triple::GNUSF:
  case toolchain::Triple::MuslSF:
    return IsLA32 ? "ilp32s" : "lp64s";
  case toolchain::Triple::GNUF32:
  case toolchain::Triple::MuslF32:
    return IsLA32 ? "ilp32f" : "lp64f";
  case toolchain::Triple::GNUF64:
    // This was originally permitted (and indeed the canonical way) to
    // represent the {ILP32,LP64}D ABIs, but in Feb 2023 Loongson decided to
    // drop the explicit suffix in favor of unmarked `-gnu` for the
    // "general-purpose" ABIs, among other non-technical reasons.
    //
    // The spec change did not mention whether existing usages of "gnuf64"
    // shall remain valid or not, so we are going to continue recognizing it
    // for some time, until it is clear that everyone else has migrated away
    // from it.
    [[fallthrough]];
  case toolchain::Triple::GNU:
  default:
    return IsLA32 ? "ilp32d" : "lp64d";
  }
}

void loongarch::getLoongArchTargetFeatures(const Driver &D,
                                           const toolchain::Triple &Triple,
                                           const ArgList &Args,
                                           std::vector<StringRef> &Features) {
  // Enable the `lsx` feature on 64-bit LoongArch by default.
  if (Triple.isLoongArch64() &&
      (!Args.hasArgNoClaim(language::Core::driver::options::OPT_march_EQ)))
    Features.push_back("+lsx");

  // FIXME: Now we must use -mrelax to enable relax, maybe -mrelax will be set
  // as default in the future.
  if (const Arg *A =
          Args.getLastArg(options::OPT_mrelax, options::OPT_mno_relax)) {
    if (A->getOption().matches(options::OPT_mrelax)) {
      Features.push_back("+relax");
      // -gsplit-dwarf -mrelax requires DW_AT_high_pc/DW_AT_ranges/... indexing
      // into .debug_addr, which is currently not implemented.
      Arg *A;
      if (getDebugFissionKind(D, Args, A) != DwarfFissionKind::None)
        D.Diag(
            language::Core::diag::err_drv_loongarch_unsupported_with_linker_relaxation)
            << A->getAsString(Args);
    } else {
      Features.push_back("-relax");
    }
  }

  std::string ArchName;
  const Arg *MArch = Args.getLastArg(options::OPT_march_EQ);
  if (MArch)
    ArchName = MArch->getValue();
  ArchName = postProcessTargetCPUString(ArchName, Triple);
  toolchain::LoongArch::getArchFeatures(ArchName, Features);
  if (MArch && StringRef(MArch->getValue()) == "native")
    for (auto &F : toolchain::sys::getHostCPUFeatures())
      Features.push_back(
          Args.MakeArgString((F.second ? "+" : "-") + F.first()));

  // Select floating-point features determined by -mdouble-float,
  // -msingle-float, -msoft-float and -mfpu.
  // Note: -m*-float wins any other options.
  if (const Arg *A = Args.getLastArg(options::OPT_mdouble_float,
                                     options::OPT_msingle_float,
                                     options::OPT_msoft_float)) {
    if (A->getOption().matches(options::OPT_mdouble_float)) {
      Features.push_back("+f");
      Features.push_back("+d");
    } else if (A->getOption().matches(options::OPT_msingle_float)) {
      Features.push_back("+f");
      Features.push_back("-d");
      Features.push_back("-lsx");
    } else /*Soft-float*/ {
      Features.push_back("-f");
      Features.push_back("-d");
      Features.push_back("-lsx");
    }
  } else if (const Arg *A = Args.getLastArg(options::OPT_mfpu_EQ)) {
    StringRef FPU = A->getValue();
    if (FPU == "64") {
      Features.push_back("+f");
      Features.push_back("+d");
    } else if (FPU == "32") {
      Features.push_back("+f");
      Features.push_back("-d");
      Features.push_back("-lsx");
    } else if (FPU == "0" || FPU == "none") {
      Features.push_back("-f");
      Features.push_back("-d");
      Features.push_back("-lsx");
    } else {
      D.Diag(diag::err_drv_loongarch_invalid_mfpu_EQ) << FPU;
    }
  }

  // Accept but warn about these TargetSpecific options.
  if (Arg *A = Args.getLastArgNoClaim(options::OPT_mabi_EQ))
    A->ignoreTargetSpecific();
  if (Arg *A = Args.getLastArgNoClaim(options::OPT_mfpu_EQ))
    A->ignoreTargetSpecific();
  if (Arg *A = Args.getLastArgNoClaim(options::OPT_msimd_EQ))
    A->ignoreTargetSpecific();

  // Select lsx/lasx feature determined by -msimd=.
  // Option -msimd= precedes -m[no-]lsx and -m[no-]lasx.
  if (const Arg *A = Args.getLastArg(options::OPT_msimd_EQ)) {
    StringRef MSIMD = A->getValue();
    if (MSIMD == "lsx") {
      // Option -msimd=lsx depends on 64-bit FPU.
      // -m*-float and -mfpu=none/0/32 conflict with -msimd=lsx.
      if (toolchain::is_contained(Features, "-d"))
        D.Diag(diag::err_drv_loongarch_wrong_fpu_width) << /*LSX*/ 0;
      else
        Features.push_back("+lsx");
    } else if (MSIMD == "lasx") {
      // Option -msimd=lasx depends on 64-bit FPU and LSX.
      // -m*-float, -mfpu=none/0/32 and -mno-lsx conflict with -msimd=lasx.
      if (toolchain::is_contained(Features, "-d"))
        D.Diag(diag::err_drv_loongarch_wrong_fpu_width) << /*LASX*/ 1;
      else if (toolchain::is_contained(Features, "-lsx"))
        D.Diag(diag::err_drv_loongarch_invalid_simd_option_combination);

      // The command options do not contain -mno-lasx.
      if (!Args.getLastArg(options::OPT_mno_lasx)) {
        Features.push_back("+lsx");
        Features.push_back("+lasx");
      }
    } else if (MSIMD == "none") {
      if (toolchain::is_contained(Features, "+lsx"))
        Features.push_back("-lsx");
      if (toolchain::is_contained(Features, "+lasx"))
        Features.push_back("-lasx");
    } else {
      D.Diag(diag::err_drv_loongarch_invalid_msimd_EQ) << MSIMD;
    }
  }

  // Select lsx feature determined by -m[no-]lsx.
  if (const Arg *A = Args.getLastArg(options::OPT_mlsx, options::OPT_mno_lsx)) {
    // LSX depends on 64-bit FPU.
    // -m*-float and -mfpu=none/0/32 conflict with -mlsx.
    if (A->getOption().matches(options::OPT_mlsx)) {
      if (toolchain::find(Features, "-d") != Features.end())
        D.Diag(diag::err_drv_loongarch_wrong_fpu_width) << /*LSX*/ 0;
      else /*-mlsx*/
        Features.push_back("+lsx");
    } else /*-mno-lsx*/ {
      Features.push_back("-lsx");
      Features.push_back("-lasx");
    }
  }

  // Select lasx feature determined by -m[no-]lasx.
  if (const Arg *A =
          Args.getLastArg(options::OPT_mlasx, options::OPT_mno_lasx)) {
    // LASX depends on 64-bit FPU and LSX.
    // -mno-lsx conflicts with -mlasx.
    if (A->getOption().matches(options::OPT_mlasx)) {
      if (toolchain::find(Features, "-d") != Features.end())
        D.Diag(diag::err_drv_loongarch_wrong_fpu_width) << /*LASX*/ 1;
      else { /*-mlasx*/
        Features.push_back("+lsx");
        Features.push_back("+lasx");
      }
    } else /*-mno-lasx*/
      Features.push_back("-lasx");
  }

  AddTargetFeature(Args, Features, options::OPT_mno_strict_align,
                   options::OPT_mstrict_align, "ual");
  AddTargetFeature(Args, Features, options::OPT_mno_strict_align,
                   options::OPT_mstrict_align, "ual");
  AddTargetFeature(Args, Features, options::OPT_mfrecipe,
                   options::OPT_mno_frecipe, "frecipe");
  AddTargetFeature(Args, Features, options::OPT_mlam_bh,
                   options::OPT_mno_lam_bh, "lam-bh");
  AddTargetFeature(Args, Features, options::OPT_mlamcas,
                   options::OPT_mno_lamcas, "lamcas");
  AddTargetFeature(Args, Features, options::OPT_mld_seq_sa,
                   options::OPT_mno_ld_seq_sa, "ld-seq-sa");
  AddTargetFeature(Args, Features, options::OPT_mdiv32,
                   options::OPT_mno_div32, "div32");
  AddTargetFeature(Args, Features, options::OPT_mscq, options::OPT_mno_scq,
                   "scq");
}

std::string loongarch::postProcessTargetCPUString(const std::string &CPU,
                                                  const toolchain::Triple &Triple) {
  std::string CPUString = CPU;
  if (CPUString == "native") {
    CPUString = toolchain::sys::getHostCPUName();
    if (CPUString == "generic")
      CPUString = toolchain::LoongArch::getDefaultArch(Triple.isLoongArch64());
  }
  if (CPUString.empty())
    CPUString = toolchain::LoongArch::getDefaultArch(Triple.isLoongArch64());
  return CPUString;
}

std::string loongarch::getLoongArchTargetCPU(const toolchain::opt::ArgList &Args,
                                             const toolchain::Triple &Triple) {
  std::string CPU;
  std::string Arch;
  // If we have -march, use that.
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    Arch = A->getValue();
    if (Arch == "la64v1.0" || Arch == "la64v1.1")
      CPU = toolchain::LoongArch::getDefaultArch(Triple.isLoongArch64());
    else
      CPU = Arch;
  }
  return postProcessTargetCPUString(CPU, Triple);
}
