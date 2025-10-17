/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

//===-- ARMMCAsmInfo.cpp - ARM asm properties -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARMMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARMMCAsmInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

cl::opt<bool>
EnableARMEHABI("arm-enable-ehabi", cl::Hidden,
  cl::desc("Generate ARM EHABI tables"),
  cl::init(false));


void ARMMCAsmInfoDarwin::anchor() { }

ARMMCAsmInfoDarwin::ARMMCAsmInfoDarwin() {
  Data64bitsDirective = 0;
  CommentString = "@";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";
  UseDataRegionDirectives = true;

  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::SjLj;
}

void ARMELFMCAsmInfo::anchor() { }

ARMELFMCAsmInfo::ARMELFMCAsmInfo() {
  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  Data64bitsDirective = 0;
  CommentString = "@";
  PrivateGlobalPrefix = ".L";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";

  WeakRefDirective = "\t.weak\t";

  HasLEB128 = true;
  SupportsDebugInformation = true;

  // Exceptions handling
  if (EnableARMEHABI)
    ExceptionsType = ExceptionHandling::ARM;
}
