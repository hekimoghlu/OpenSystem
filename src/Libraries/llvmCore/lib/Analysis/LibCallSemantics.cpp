/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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

//===- LibCallSemantics.cpp - Describe library semantics ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements interfaces that can be used to describe language
// specific runtime library interfaces (e.g. libc, libm, etc) to LLVM
// optimizers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Function.h"
using namespace llvm;

/// getMap - This impl pointer in ~LibCallInfo is actually a StringMap.  This
/// helper does the cast.
static StringMap<const LibCallFunctionInfo*> *getMap(void *Ptr) {
  return static_cast<StringMap<const LibCallFunctionInfo*> *>(Ptr);
}

LibCallInfo::~LibCallInfo() {
  delete getMap(Impl);
}

const LibCallLocationInfo &LibCallInfo::getLocationInfo(unsigned LocID) const {
  // Get location info on the first call.
  if (NumLocations == 0)
    NumLocations = getLocationInfo(Locations);
  
  assert(LocID < NumLocations && "Invalid location ID!");
  return Locations[LocID];
}


/// getFunctionInfo - Return the LibCallFunctionInfo object corresponding to
/// the specified function if we have it.  If not, return null.
const LibCallFunctionInfo *
LibCallInfo::getFunctionInfo(const Function *F) const {
  StringMap<const LibCallFunctionInfo*> *Map = getMap(Impl);
  
  /// If this is the first time we are querying for this info, lazily construct
  /// the StringMap to index it.
  if (Map == 0) {
    Impl = Map = new StringMap<const LibCallFunctionInfo*>();
    
    const LibCallFunctionInfo *Array = getFunctionInfoArray();
    if (Array == 0) return 0;
    
    // We now have the array of entries.  Populate the StringMap.
    for (unsigned i = 0; Array[i].Name; ++i)
      (*Map)[Array[i].Name] = Array+i;
  }
  
  // Look up this function in the string map.
  return Map->lookup(F->getName());
}

