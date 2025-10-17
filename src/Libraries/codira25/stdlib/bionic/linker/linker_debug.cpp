/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#include "linker_debug.h"

#include <unistd.h>

#include <android-base/strings.h>

LinkerDebugConfig g_linker_debug_config;

void init_LD_DEBUG(const std::string& value) {
  if (value.empty()) return;
  std::vector<std::string> options = android::base::Split(value, ",");
  for (const auto& o : options) {
    if (o == "calls") g_linker_debug_config.calls = true;
    else if (o == "cfi") g_linker_debug_config.cfi = true;
    else if (o == "dynamic") g_linker_debug_config.dynamic = true;
    else if (o == "lookup") g_linker_debug_config.lookup = true;
    else if (o == "props") g_linker_debug_config.props = true;
    else if (o == "reloc") g_linker_debug_config.reloc = true;
    else if (o == "statistics") g_linker_debug_config.statistics = true;
    else if (o == "timing") g_linker_debug_config.timing = true;
    else if (o == "all") {
      g_linker_debug_config.calls = true;
      g_linker_debug_config.cfi = true;
      g_linker_debug_config.dynamic = true;
      g_linker_debug_config.lookup = true;
      g_linker_debug_config.props = true;
      g_linker_debug_config.reloc = true;
      g_linker_debug_config.statistics = true;
      g_linker_debug_config.timing = true;
    } else {
      __linker_error("$LD_DEBUG is a comma-separated list of:\n"
                     "\n"
                     "  calls       ctors/dtors/ifuncs\n"
                     "  cfi         control flow integrity messages\n"
                     "  dynamic     dynamic section processing\n"
                     "  lookup      symbol lookup\n"
                     "  props       ELF property processing\n"
                     "  reloc       relocation resolution\n"
                     "  statistics  relocation statistics\n"
                     "  timing      timing information\n"
                     "\n"
                     "or 'all' for all of the above.\n");
    }
  }
  if (g_linker_debug_config.calls || g_linker_debug_config.cfi ||
      g_linker_debug_config.dynamic || g_linker_debug_config.lookup ||
      g_linker_debug_config.props || g_linker_debug_config.reloc ||
      g_linker_debug_config.statistics || g_linker_debug_config.timing) {
    g_linker_debug_config.any = true;
  }
}

static void linker_log_va_list(int prio, const char* fmt, va_list ap) {
  va_list ap2;
  va_copy(ap2, ap);
  async_safe_format_log_va_list(prio, "linker", fmt, ap2);
  va_end(ap2);

  async_safe_format_fd_va_list(STDERR_FILENO, fmt, ap);
  write(STDERR_FILENO, "\n", 1);
}

void __linker_log(int prio, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  linker_log_va_list(prio, fmt, ap);
  va_end(ap);
}

void __linker_error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  linker_log_va_list(ANDROID_LOG_FATAL, fmt, ap);
  va_end(ap);

  _exit(EXIT_FAILURE);
}
