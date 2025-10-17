/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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
#include "linker_debuggerd.h"

#include "debuggerd/handler.h"
#include "private/bionic_globals.h"

#include "linker_gdb_support.h"

#if defined(__ANDROID_APEX__)
static debugger_process_info get_process_info() {
  return {
      .abort_msg = __libc_shared_globals()->abort_msg,
      .fdsan_table = &__libc_shared_globals()->fd_table,
      .gwp_asan_state = __libc_shared_globals()->gwp_asan_state,
      .gwp_asan_metadata = __libc_shared_globals()->gwp_asan_metadata,
      .scudo_stack_depot = __libc_shared_globals()->scudo_stack_depot,
      .scudo_region_info = __libc_shared_globals()->scudo_region_info,
      .scudo_ring_buffer = __libc_shared_globals()->scudo_ring_buffer,
      .scudo_ring_buffer_size = __libc_shared_globals()->scudo_ring_buffer_size,
      .scudo_stack_depot_size = __libc_shared_globals()->scudo_stack_depot_size,
      .crash_detail_page = __libc_shared_globals()->crash_detail_page,
  };
}

static gwp_asan_callbacks_t get_gwp_asan_callbacks() {
  return {
      .debuggerd_needs_gwp_asan_recovery =
          __libc_shared_globals()->debuggerd_needs_gwp_asan_recovery,
      .debuggerd_gwp_asan_pre_crash_report =
          __libc_shared_globals()->debuggerd_gwp_asan_pre_crash_report,
      .debuggerd_gwp_asan_post_crash_report =
          __libc_shared_globals()->debuggerd_gwp_asan_post_crash_report,
  };
}
#endif

void linker_debuggerd_init() {
  // There may be a version mismatch between the bootstrap linker and the crash_dump in the APEX,
  // so don't pass in any process info from the bootstrap linker.
  debuggerd_callbacks_t callbacks = {
#if defined(__ANDROID_APEX__)
    .get_process_info = get_process_info,
    .get_gwp_asan_callbacks = get_gwp_asan_callbacks,
#endif
    .post_dump = notify_gdb_of_libraries,
  };
  debuggerd_init(&callbacks);
}
