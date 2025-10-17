/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

//===-- DIContext.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DIContext.h"
#include "DWARFContext.h"
using namespace llvm;

DIContext::~DIContext() {}

DIContext *DIContext::getDWARFContext(bool isLittleEndian,
                                      StringRef infoSection,
                                      StringRef abbrevSection,
                                      StringRef aRangeSection,
                                      StringRef lineSection,
                                      StringRef stringSection,
                                      StringRef rangeSection) {
  return new DWARFContextInMemory(isLittleEndian, infoSection, abbrevSection,
                                  aRangeSection, lineSection, stringSection,
                                  rangeSection);
}
