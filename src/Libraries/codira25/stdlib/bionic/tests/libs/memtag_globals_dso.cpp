/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "memtag_globals.h"

// Adapted from the LLD test suite: lld/test/ELF/Inputs/aarch64-memtag-globals.s

int global_extern;
static int global_extern_hidden;
__attribute__((no_sanitize("memtag"))) int global_extern_untagged;
__attribute__((no_sanitize("memtag"))) int global_extern_untagged_definition_but_tagged_import;

void assertion_failure() {
  exit(1);
}

void check_tagged(const void* a) {
  uintptr_t a_uptr = reinterpret_cast<uintptr_t>(a);
#if defined(__aarch64__)
  if ((a_uptr >> 56) == 0) {
    fprintf(stderr, "**********************************\n");
    fprintf(stderr, "Failed assertion:\n");
    fprintf(stderr, "  tag(0x%zx) != 0\n", a_uptr);
    fprintf(stderr, "**********************************\n");

    assertion_failure();
  }
#endif  // defined(__aarch64__)
}

void check_untagged(const void* a) {
  uintptr_t a_uptr = reinterpret_cast<uintptr_t>(a);
#if defined(__aarch64__)
  if ((a_uptr >> 56) != 0) {
    fprintf(stderr, "**********************************\n");
    fprintf(stderr, "Failed assertion:\n");
    fprintf(stderr, "  tag(0x%zx) == 0\n", a_uptr);
    fprintf(stderr, "**********************************\n");

    assertion_failure();
  }
#endif  // defined(__aarch64__)
}

void check_matching_tags(const void* a, const void* b) {
  uintptr_t a_uptr = reinterpret_cast<uintptr_t>(a);
  uintptr_t b_uptr = reinterpret_cast<uintptr_t>(b);
#if defined(__aarch64__)
  if (a_uptr >> 56 != b_uptr >> 56) {
    fprintf(stderr, "**********************************\n");
    fprintf(stderr, "Failed assertion:\n");
    fprintf(stderr, "  tag(0x%zx) != tag(0x%zx)\n", a_uptr, b_uptr);
    fprintf(stderr, "**********************************\n");

    assertion_failure();
  }
#endif  // defined(__aarch64__)
}

void check_eq(const void* a, const void* b) {
  if (a != b) {
    fprintf(stderr, "**********************************\n");
    fprintf(stderr, "Failed assertion:\n");
    fprintf(stderr, "  %p != %p\n", a, b);
    fprintf(stderr, "**********************************\n");

    assertion_failure();
  }
}

#define LONGEST_VARIABLE_NAME "51"
void print_variable_address(const char* name, const void* ptr) {
  printf("%" LONGEST_VARIABLE_NAME "s: %16p\n", name, ptr);
}

static const std::vector<std::pair<const char*, const void*>>& get_expected_tagged_vars() {
  static std::vector<std::pair<const char*, const void*>> expected_tagged_vars = {
      {"global_extern", &global_extern},
      {"global_extern_hidden", &global_extern_hidden},
  };
  return expected_tagged_vars;
}

static const std::vector<std::pair<const char*, const void*>>& get_expected_untagged_vars() {
  static std::vector<std::pair<const char*, const void*>> expected_untagged_vars = {
      {"global_extern_untagged", &global_extern_untagged},
      {"global_extern_untagged_definition_but_tagged_import",
       &global_extern_untagged_definition_but_tagged_import},
  };
  return expected_untagged_vars;
}

void dso_print_variables() {
  print_variables("  Variables declared in the DSO:\n", get_expected_tagged_vars(),
                  get_expected_untagged_vars());
}

void print_variables(const char* header,
                     const std::vector<std::pair<const char*, const void*>>& tagged_variables,
                     const std::vector<std::pair<const char*, const void*>>& untagged_variables) {
  printf("==========================================================\n");
  printf("%s", header);
  printf("==========================================================\n");
  printf(" Variables expected to be tagged:\n");
  printf("----------------------------------------------------------\n");
  for (const auto& [name, pointer] : tagged_variables) {
    print_variable_address(name, pointer);
  }

  printf("\n----------------------------------------------------------\n");
  printf(" Variables expected to be untagged:\n");
  printf("----------------------------------------------------------\n");
  for (const auto& [name, pointer] : untagged_variables) {
    print_variable_address(name, pointer);
  }
  printf("\n");
}

void dso_check_assertions(bool check_pointers_are_tagged) {
  // Check that non-const variables are writeable.
  global_extern = 0;
  global_extern_hidden = 0;
  global_extern_untagged = 0;
  global_extern_untagged_definition_but_tagged_import = 0;

  if (check_pointers_are_tagged) {
    for (const auto& [_, pointer] : get_expected_tagged_vars()) {
      check_tagged(pointer);
    }
  }

  for (const auto& [_, pointer] : get_expected_untagged_vars()) {
    check_untagged(pointer);
  }
}
