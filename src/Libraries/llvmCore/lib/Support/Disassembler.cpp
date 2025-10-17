/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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

//===- lib/Support/Disassembler.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the necessary glue to call external disassembler
// libraries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/Disassembler.h"

#include <cassert>
#include <iomanip>
#include <string>
#include <sstream>

#if USE_UDIS86
#include <udis86.h>
#endif

using namespace llvm;

bool llvm::sys::hasDisassembler()
{
#if defined (__i386__) || defined (__amd64__) || defined (__x86_64__)
  // We have option to enable udis86 library.
# if USE_UDIS86
  return true;
#else
  return false;
#endif
#else
  return false;
#endif
}

std::string llvm::sys::disassembleBuffer(uint8_t* start, size_t length,
                                         uint64_t pc) {
  std::stringstream res;

#if (defined (__i386__) || defined (__amd64__) || defined (__x86_64__)) \
  && USE_UDIS86
  unsigned bits;
# if defined(__i386__)
  bits = 32;
# else
  bits = 64;
# endif

  ud_t ud_obj;

  ud_init(&ud_obj);
  ud_set_input_buffer(&ud_obj, start, length);
  ud_set_mode(&ud_obj, bits);
  ud_set_pc(&ud_obj, pc);
  ud_set_syntax(&ud_obj, UD_SYN_ATT);

  res << std::setbase(16)
      << std::setw(bits/4);

  while (ud_disassemble(&ud_obj)) {
    res << ud_insn_off(&ud_obj) << ":\t" << ud_insn_asm(&ud_obj) << "\n";
  }
#else
  res << "No disassembler available. See configure help for options.\n";
#endif

  return res.str();
}
