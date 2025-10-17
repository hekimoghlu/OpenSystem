/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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

//===-- DWARFDebugRangesList.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugRangeList.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void DWARFDebugRangeList::clear() {
  Offset = -1U;
  AddressSize = 0;
  Entries.clear();
}

bool DWARFDebugRangeList::extract(DataExtractor data, uint32_t *offset_ptr) {
  clear();
  if (!data.isValidOffset(*offset_ptr))
    return false;
  AddressSize = data.getAddressSize();
  if (AddressSize != 4 && AddressSize != 8)
    return false;
  Offset = *offset_ptr;
  while (true) {
    RangeListEntry entry;
    uint32_t prev_offset = *offset_ptr;
    entry.StartAddress = data.getAddress(offset_ptr);
    entry.EndAddress = data.getAddress(offset_ptr);
    // Check that both values were extracted correctly.
    if (*offset_ptr != prev_offset + 2 * AddressSize) {
      clear();
      return false;
    }
    if (entry.isEndOfListEntry())
      break;
    Entries.push_back(entry);
  }
  return true;
}

void DWARFDebugRangeList::dump(raw_ostream &OS) const {
  for (int i = 0, n = Entries.size(); i != n; ++i) {
    const char *format_str = (AddressSize == 4
                              ? "%08x %08"  PRIx64 " %08"  PRIx64 "\n"
                              : "%08x %016" PRIx64 " %016" PRIx64 "\n");
    OS << format(format_str, Offset, Entries[i].StartAddress,
                                     Entries[i].EndAddress);
  }
  OS << format("%08x <End of list>\n", Offset);
}

bool DWARFDebugRangeList::containsAddress(uint64_t BaseAddress,
                                          uint64_t Address) const {
  for (int i = 0, n = Entries.size(); i != n; ++i) {
    if (Entries[i].isBaseAddressSelectionEntry(AddressSize))
      BaseAddress = Entries[i].EndAddress;
    else if (Entries[i].containsAddress(BaseAddress, Address))
      return true;
  }
  return false;
}
