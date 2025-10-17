/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#pragma once

#include <stdarg.h>
#include <unistd.h>

#include <string>

#include <async_safe/log.h>
#include <async_safe/CHECK.h>

struct LinkerDebugConfig {
  // Set automatically if any of the more specific options are set.
  bool any;

  // Messages relating to calling ctors/dtors/ifuncs.
  bool calls;
  // Messages relating to CFI.
  bool cfi;
  // Messages relating to the dynamic section.
  bool dynamic;
  // Messages relating to symbol lookup.
  bool lookup;
  // Messages relating to relocation processing.
  bool reloc;
  // Messages relating to ELF properties.
  bool props;
  // TODO: "config" and "zip" seem likely to want to be separate?

  bool timing;
  bool statistics;
};

extern LinkerDebugConfig g_linker_debug_config;

__LIBC_HIDDEN__ void init_LD_DEBUG(const std::string& value);
__LIBC_HIDDEN__ void __linker_log(int prio, const char* fmt, ...) __printflike(2, 3);
__LIBC_HIDDEN__ void __linker_error(const char* fmt, ...) __printflike(1, 2);

#define LD_DEBUG(what, x...) \
  do { \
    if (g_linker_debug_config.what) { \
      __linker_log(ANDROID_LOG_INFO, x); \
    } \
  } while (false)
