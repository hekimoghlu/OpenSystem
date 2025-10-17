/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
// Prevent tests from being compiled with glibc because thread_properties.h
// only exists in Bionic.
#if defined(__BIONIC__)

#include <sys/thread_properties.h>

#include <assert.h>
#include <dlfcn.h>
#include <elf.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

// Helper binary to use TLS-related functions in thread_properties

// Tests __get_static_tls_bound.
thread_local int local_var;
void test_static_tls_bounds() {
  local_var = 123;
  void* start_addr = nullptr;
  void* end_addr = nullptr;

  __libc_get_static_tls_bounds(reinterpret_cast<void**>(&start_addr),
                               reinterpret_cast<void**>(&end_addr));
  assert(start_addr != nullptr);
  assert(end_addr != nullptr);

  assert(&local_var >= start_addr && &local_var < end_addr);

  printf("done_get_static_tls_bounds\n");
}

// Tests iterate_dynamic tls chunks.
// Export a var from the shared so.
__thread char large_tls_var[4 * 1024 * 1024];
// found_count  has to be Global variable so that the non-capturing lambda
// can access it.
int found_count = 0;
void test_iter_tls() {
  void* lib = dlopen("libtest_elftls_dynamic.so", RTLD_LOCAL | RTLD_NOW);
  large_tls_var[1025] = 'a';
  auto cb = +[](void* dtls_begin, void* dtls_end, size_t dso_id, void* arg) {
    if (&large_tls_var >= dtls_begin && &large_tls_var < dtls_end) ++found_count;
  };
  __libc_iterate_dynamic_tls(gettid(), cb, nullptr);

  // It should be found exactly once.
  assert(found_count == 1);
  printf("done_iterate_dynamic_tls\n");
}

void* parent_addr = nullptr;
void test_iterate_another_thread_tls() {
  large_tls_var[1025] = 'b';
  parent_addr = &large_tls_var;
  found_count = 0;

  pid_t pid = fork();
  assert(pid != -1);
  int status;
  if (pid) {
    // Parent.
    assert(pid == wait(&status));
    assert(0 == status);
  } else {
    // Child.
    pid_t parent_pid = getppid();
    assert(0 == ptrace(PTRACE_ATTACH, parent_pid));
    assert(parent_pid == waitpid(parent_pid, &status, 0));

    auto cb = +[](void* dtls_begin, void* dtls_end, size_t dso_id, void* arg) {
      if (parent_addr >= dtls_begin && parent_addr < dtls_end) ++found_count;
    };
    __libc_iterate_dynamic_tls(parent_pid, cb, nullptr);
    // It should be found exactly once.
    assert(found_count == 1);
    printf("done_iterate_another_thread_tls\n");
  }
}
int main() {
  test_static_tls_bounds();
  test_iter_tls();
  test_iterate_another_thread_tls();
  return 0;
}

#else
int main() {
  return 0;
}
#endif  // __BIONIC__
