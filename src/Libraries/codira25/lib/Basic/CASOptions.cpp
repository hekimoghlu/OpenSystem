/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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

//===--- CASOptions.cpp - CAS & caching options ---------------------------===//
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
//  This file defines the CASOptions class, which provides various
//  CAS and caching flags.
//
//===----------------------------------------------------------------------===//

#include "language/Basic/CASOptions.h"

using namespace language;

void CASOptions::enumerateCASConfigurationFlags(
      toolchain::function_ref<void(toolchain::StringRef)> Callback) const {
  if (EnableCaching) {
    Callback("-cache-compile-job");
    if (!CASOpts.CASPath.empty()) {
      Callback("-cas-path");
      Callback(CASOpts.CASPath);
    }
    if (!CASOpts.PluginPath.empty()) {
      Callback("-cas-plugin-path");
      Callback(CASOpts.PluginPath);
      for (auto Opt : CASOpts.PluginOptions) {
        Callback("-cas-plugin-option");
        Callback((toolchain::Twine(Opt.first) + "=" + Opt.second).str());
      }
    }
  }
}
