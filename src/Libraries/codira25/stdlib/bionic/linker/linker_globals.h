/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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

#include "linker_debug.h"

#include <link.h>
#include <stddef.h>

#include <string>
#include <unordered_map>

#include <async_safe/log.h>

#define DL_ERR(fmt, x...) \
    do { \
      async_safe_format_buffer(linker_get_error_buffer(), linker_get_error_buffer_size(), fmt, ##x); \
    } while (false)

#define DL_WARN(fmt, x...) \
    do { \
      async_safe_format_log(ANDROID_LOG_WARN, "linker", fmt, ##x); \
      async_safe_format_fd(2, "WARNING: linker: "); \
      async_safe_format_fd(2, fmt, ##x); \
      async_safe_format_fd(2, "\n"); \
    } while (false)

bool DL_ERROR_AFTER(int target_sdk_version, const char* fmt, ...) __printflike(2, 3);

#define DL_ERR_AND_LOG(fmt, x...) \
  do { \
    DL_ERR(fmt, ##x); \
    __linker_log(ANDROID_LOG_ERROR, fmt, ##x); \
  } while (false)

#define DL_OPEN_ERR(fmt, x...) \
  do { \
    DL_ERR(fmt, ##x); \
    LD_LOG(kLogDlopen, fmt, ##x); \
  } while (false)

#define DL_SYM_ERR(fmt, x...) \
  do { \
    DL_ERR(fmt, ##x); \
    LD_LOG(kLogDlsym, fmt, ##x); \
  } while (false)

constexpr ElfW(Versym) kVersymNotNeeded = 0;
constexpr ElfW(Versym) kVersymGlobal = 1;

// These values are used to call constructors for .init_array && .preinit_array
extern int g_argc;
extern char** g_argv;
extern char** g_envp;

struct soinfo;
struct android_namespace_t;
struct platform_properties;

extern android_namespace_t g_default_namespace;

extern std::unordered_map<uintptr_t, soinfo*> g_soinfo_handles_map;

extern platform_properties g_platform_properties;

// Error buffer "variable"
char* linker_get_error_buffer();
size_t linker_get_error_buffer_size();

class DlErrorRestorer {
 public:
  DlErrorRestorer() {
    saved_error_msg_ = linker_get_error_buffer();
  }
  ~DlErrorRestorer() {
    strlcpy(linker_get_error_buffer(), saved_error_msg_.c_str(), linker_get_error_buffer_size());
  }
 private:
  std::string saved_error_msg_;
};

__LIBC_HIDDEN__ extern bool g_is_ldd;
__LIBC_HIDDEN__ extern pthread_mutex_t g_dl_mutex;
