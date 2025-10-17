/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#include <sys/sysinfo.h>

#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "platform/bionic/page.h"
#include "private/ScopedReaddir.h"
#include "private/get_cpu_count_from_string.h"

int __get_cpu_count(const char* sys_file) {
  int cpu_count = 1;
  FILE* fp = fopen(sys_file, "re");
  if (fp != nullptr) {
    char* line = nullptr;
    size_t allocated_size = 0;
    if (getline(&line, &allocated_size, fp) != -1) {
      cpu_count = GetCpuCountFromString(line);
    }
    free(line);
    fclose(fp);
  }
  return cpu_count;
}

int get_nprocs_conf() {
  // It's unclear to me whether this is intended to be "possible" or "present",
  // but on mobile they're unlikely to differ.
  return __get_cpu_count("/sys/devices/system/cpu/possible");
}

int get_nprocs() {
  return __get_cpu_count("/sys/devices/system/cpu/online");
}

long get_phys_pages() {
  struct sysinfo si;
  sysinfo(&si);
  return (static_cast<int64_t>(si.totalram) * si.mem_unit) / page_size();
}

long get_avphys_pages() {
  struct sysinfo si;
  sysinfo(&si);
  return ((static_cast<int64_t>(si.freeram) + si.bufferram) * si.mem_unit) / page_size();
}
