/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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

//===- llvm/PassRegistry.h - Pass Information Registry ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines PassRegistry, a class that is used in the initialization
// and registration of passes.  At application startup, passes are registered
// with the PassRegistry, which is later provided to the PassManager for 
// dependency resolution and similar tasks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSREGISTRY_H
#define LLVM_PASSREGISTRY_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class PassInfo;
struct PassRegistrationListener;

/// PassRegistry - This class manages the registration and intitialization of
/// the pass subsystem as application startup, and assists the PassManager
/// in resolving pass dependencies.
/// NOTE: PassRegistry is NOT thread-safe.  If you want to use LLVM on multiple
/// threads simultaneously, you will need to use a separate PassRegistry on
/// each thread.
class PassRegistry {
  mutable void *pImpl;
  void *getImpl() const;
   
public:
  PassRegistry() : pImpl(0) { }
  ~PassRegistry();
  
  /// getPassRegistry - Access the global registry object, which is 
  /// automatically initialized at application launch and destroyed by
  /// llvm_shutdown.
  static PassRegistry *getPassRegistry();
  
  /// getPassInfo - Look up a pass' corresponding PassInfo, indexed by the pass'
  /// type identifier (&MyPass::ID).
  const PassInfo *getPassInfo(const void *TI) const;
  
  /// getPassInfo - Look up a pass' corresponding PassInfo, indexed by the pass'
  /// argument string.
  const PassInfo *getPassInfo(StringRef Arg) const;
  
  /// registerPass - Register a pass (by means of its PassInfo) with the 
  /// registry.  Required in order to use the pass with a PassManager.
  void registerPass(const PassInfo &PI, bool ShouldFree = false);
  
  /// registerPass - Unregister a pass (by means of its PassInfo) with the 
  /// registry.
  void unregisterPass(const PassInfo &PI);
  
  /// registerAnalysisGroup - Register an analysis group (or a pass implementing
  // an analysis group) with the registry.  Like registerPass, this is required 
  // in order for a PassManager to be able to use this group/pass.
  void registerAnalysisGroup(const void *InterfaceID, const void *PassID,
                             PassInfo& Registeree, bool isDefault,
                             bool ShouldFree = false);
  
  /// enumerateWith - Enumerate the registered passes, calling the provided
  /// PassRegistrationListener's passEnumerate() callback on each of them.
  void enumerateWith(PassRegistrationListener *L);
  
  /// addRegistrationListener - Register the given PassRegistrationListener
  /// to receive passRegistered() callbacks whenever a new pass is registered.
  void addRegistrationListener(PassRegistrationListener *L);
  
  /// removeRegistrationListener - Unregister a PassRegistrationListener so that
  /// it no longer receives passRegistered() callbacks.
  void removeRegistrationListener(PassRegistrationListener *L);
};

}

#endif
