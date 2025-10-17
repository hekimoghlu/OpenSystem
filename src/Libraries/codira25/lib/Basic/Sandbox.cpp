/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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

#include "language/Basic/Sandbox.h"
#include "language/Basic/Toolchain.h"
#include "language/Basic/StringExtras.h"
#include "toolchain/ADT/SmallString.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

using namespace language;
using namespace Sandbox;

#if defined(__APPLE__) && TARGET_OS_OSX
static StringRef sandboxProfile(toolchain::BumpPtrAllocator &Alloc) {
  toolchain::SmallString<256> contents;
  contents += "(version 1)\n";

  // Deny everything by default.
  contents += "(deny default)\n";

  // Import the system sandbox profile.
  contents += "(import \"system.sb\")\n";

  // Allow reading file metadata of any files.
  contents += "(allow file-read-metadata)\n";

  // Allow reading dylibs.
  contents += "(allow file-read* (regex #\"\\.dylib$\"))\n";

  // This is required to launch any processes (execve(2)).
  contents += "(allow process-exec*)\n";

  return NullTerminatedStringRef(StringRef(contents), Alloc);
}
#endif

bool language::Sandbox::apply(toolchain::SmallVectorImpl<toolchain::StringRef> &command,
                           toolchain::BumpPtrAllocator &Alloc) {
#if defined(__APPLE__) && TARGET_OS_OSX
  auto profile = sandboxProfile(Alloc);
  command.insert(command.begin(), {"/usr/bin/sandbox-exec", "-p", profile});
#endif
  return false;
}
