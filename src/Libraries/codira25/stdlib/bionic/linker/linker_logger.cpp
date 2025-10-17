/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
#include "linker_logger.h"

#include <string.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <string>
#include <vector>

#include <async_safe/log.h>

#include "android-base/strings.h"
#include "private/CachedProperty.h"

LinkerLogger g_linker_logger;

static uint32_t ParseProperty(const std::string& value) {
  if (value.empty()) {
    return 0;
  }

  std::vector<std::string> options = android::base::Split(value, ",");

  uint32_t flags = 0;

  for (const auto& o : options) {
    if (o == "dlerror") {
      flags |= kLogErrors;
    } else if (o == "dlopen") {
      flags |= kLogDlopen;
    } else if (o == "dlsym") {
      flags |= kLogDlsym;
    } else {
      async_safe_format_log(ANDROID_LOG_WARN, "linker", "Ignoring unknown debug.ld option \"%s\"",
                            o.c_str());
    }
  }

  return flags;
}

static void GetAppSpecificProperty(char* buffer) {
  // Get process basename.
  const char* process_name_start = basename(g_argv[0]);

  // Remove ':' and everything after it. This is the naming convention for
  // services: https://developer.android.com/guide/components/services.html
  const char* process_name_end = strchr(process_name_start, ':');

  std::string process_name = (process_name_end != nullptr) ?
                             std::string(process_name_start, (process_name_end - process_name_start)) :
                             std::string(process_name_start);

  std::string property_name = std::string("debug.ld.app.") + process_name;
  __system_property_get(property_name.c_str(), buffer);
}

void LinkerLogger::ResetState() {
  // The most likely scenario app is not debuggable and
  // is running on a user build, in which case logging is disabled.
  if (prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) == 0) {
    return;
  }

  flags_ = 0;

  // For logging, check the flag applied to all processes first.
  static CachedProperty debug_ld_all("debug.ld.all");
  flags_ |= ParseProperty(debug_ld_all.Get());

  // Safeguard against a NULL g_argv. Ignore processes started without argv (http://b/33276926).
  if (g_argv == nullptr || g_argv[0] == nullptr) {
    return;
  }

  // Otherwise check the app-specific property too.
  // We can't easily cache the property here because argv[0] changes.
  char debug_ld_app[PROP_VALUE_MAX] = {};
  GetAppSpecificProperty(debug_ld_app);
  flags_ |= ParseProperty(debug_ld_app);
}

void LinkerLogger::Log(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  async_safe_format_log_va_list(ANDROID_LOG_DEBUG, "linker", format, ap);
  va_end(ap);
}
