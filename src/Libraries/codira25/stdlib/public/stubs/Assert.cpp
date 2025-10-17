/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

//===--- Assert.cpp - Assertion failure reporting -------------------------===//
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

#include "language/Runtime/Config.h"
#include "language/Runtime/Debug.h"
#include "language/Runtime/Portability.h"
#include "language/shims/AssertionReporting.h"
#include <cstdarg>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

using namespace language;

static void logPrefixAndMessageToDebugger(
    const unsigned char *prefix, int prefixLength,
    const unsigned char *message, int messageLength
) {
  if (!_language_shouldReportFatalErrorsToDebugger())
    return;

  char *debuggerMessage;
  if (messageLength) {
    language_asprintf(&debuggerMessage, "%.*s: %.*s", prefixLength, prefix,
        messageLength, message);
  } else {
    language_asprintf(&debuggerMessage, "%.*s", prefixLength, prefix);
  }
  _language_reportToDebugger(RuntimeErrorFlagFatal, debuggerMessage);
  free(debuggerMessage);
}

void _language_stdlib_reportFatalErrorInFile(
    const unsigned char *prefix, int prefixLength,
    const unsigned char *message, int messageLength,
    const unsigned char *file, int fileLength,
    uint32_t line,
    uint32_t flags
) {
  char *log;
  language_asprintf(
      &log, "%.*s:%" PRIu32 ": %.*s%s%.*s\n",
      fileLength, file,
      line,
      prefixLength, prefix,
      (messageLength > 0 ? ": " : ""),
      messageLength, message);

  language_reportError(flags, log);
  free(log);

  logPrefixAndMessageToDebugger(prefix, prefixLength, message, messageLength);
}

void _language_stdlib_reportFatalError(
    const unsigned char *prefix, int prefixLength,
    const unsigned char *message, int messageLength,
    uint32_t flags
) {
  char *log;
  language_asprintf(
      &log, "%.*s: %.*s\n",
      prefixLength, prefix,
      messageLength, message);

  language_reportError(flags, log);
  free(log);

  logPrefixAndMessageToDebugger(prefix, prefixLength, message, messageLength);
}

void _language_stdlib_reportUnimplementedInitializerInFile(
    const unsigned char *className, int classNameLength,
    const unsigned char *initName, int initNameLength,
    const unsigned char *file, int fileLength,
    uint32_t line, uint32_t column,
    uint32_t flags
) {
  char *log;
  language_asprintf(
      &log,
      "%.*s:%" PRIu32 ": Fatal error: Use of unimplemented "
      "initializer '%.*s' for class '%.*s'\n",
      fileLength, file,
      line,
      initNameLength, initName,
      classNameLength, className);

  language_reportError(flags, log);
  free(log);
}

void _language_stdlib_reportUnimplementedInitializer(
    const unsigned char *className, int classNameLength,
    const unsigned char *initName, int initNameLength,
    uint32_t flags
) {
  char *log;
  language_asprintf(
      &log,
      "Fatal error: Use of unimplemented "
      "initializer '%.*s' for class '%.*s'\n",
      initNameLength, initName,
      classNameLength, className);

  language_reportError(flags, log);
  free(log);
}

