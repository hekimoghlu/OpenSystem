/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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

#include <android/dlext.h>

#include <unordered_map>
#include <vector>

#include "linker_namespaces.h"
#include "linker_soinfo.h"

class ProtectedDataGuard {
 public:
  ProtectedDataGuard();
  ~ProtectedDataGuard();

 private:
  void protect_data(int protection);
  static size_t ref_count_;
};

class ElfReader;

void init_sanitizer_mode(const char *interp);
std::vector<android_namespace_t*> init_default_namespaces(const char* executable_path);
soinfo* soinfo_alloc(android_namespace_t* ns, const char* name,
                     const struct stat* file_stat, off64_t file_offset,
                     uint32_t rtld_flags);

bool find_libraries(android_namespace_t* ns,
                    soinfo* start_with,
                    const char* const library_names[],
                    size_t library_names_count,
                    soinfo* soinfos[],
                    std::vector<soinfo*>* ld_preloads,
                    size_t ld_preloads_count,
                    int rtld_flags,
                    const android_dlextinfo* extinfo,
                    bool add_as_children,
                    std::vector<android_namespace_t*>* namespaces = nullptr);

void solist_add_soinfo(soinfo* si);
bool solist_remove_soinfo(soinfo* si);
soinfo* solist_get_head();
soinfo* solist_get_somain();
soinfo* solist_get_vdso();

void linker_memcpy(void* dst, const void* src, size_t n);
