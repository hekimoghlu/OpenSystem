/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 27, 2023.
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

//===-- MCAsmInfoCOFF.h - COFF asm properties -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COFF_TARGET_ASM_INFO_H
#define LLVM_COFF_TARGET_ASM_INFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class MCAsmInfoCOFF : public MCAsmInfo {
    virtual void anchor();
  protected:
    explicit MCAsmInfoCOFF();
  };

  class MCAsmInfoMicrosoft : public MCAsmInfoCOFF {
    virtual void anchor();
  protected:
    explicit MCAsmInfoMicrosoft();
  };

  class MCAsmInfoGNUCOFF : public MCAsmInfoCOFF {
    virtual void anchor();
  protected:
    explicit MCAsmInfoGNUCOFF();
  };
}


#endif // LLVM_COFF_TARGET_ASM_INFO_H
