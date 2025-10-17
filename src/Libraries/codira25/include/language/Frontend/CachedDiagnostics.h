/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

//===--- CachedDiagnostics.h - Cached Diagnostics ---------------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
//  This file defines the CachedDiagnosticConsumer class, which
//  caches the diagnostics which can be replayed with other DiagnosticConumers.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CACHEDDIAGNOSTICS_H
#define LANGUAGE_CACHEDDIAGNOSTICS_H

#include "toolchain/Support/Error.h"

namespace language {

class CompilerInstance;
class DiagnosticEngine;
class SourceManager;
class FrontendInputsAndOutputs;

class CachingDiagnosticsProcessor {
public:
  CachingDiagnosticsProcessor(CompilerInstance &Instance);
  ~CachingDiagnosticsProcessor();

  /// Start capturing all the diagnostics from DiagnosticsEngine.
  void startDiagnosticCapture();
  /// End capturing all the diagnostics from DiagnosticsEngine.
  void endDiagnosticCapture();

  /// Emit serialized diagnostics into output stream.
  toolchain::Error serializeEmittedDiagnostics(toolchain::raw_ostream &os);

  /// Used to replay the previously cached diagnostics, after a cache hit.
  toolchain::Error replayCachedDiagnostics(toolchain::StringRef Buffer);

private:
  class Implementation;
  Implementation& Impl;
};

}

#endif
