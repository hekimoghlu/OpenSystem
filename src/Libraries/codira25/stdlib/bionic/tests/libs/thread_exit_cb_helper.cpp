/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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

#include <stdio.h>
#include <sys/thread_properties.h>

// Helper binary for testing thread_exit_cb registration.

void exit_cb_1() {
  printf("exit_cb_1 called ");
}

void exit_cb_2() {
  printf("exit_cb_2 called ");
}

void exit_cb_3() {
  printf("exit_cb_3 called");
}

void test_register_thread_exit_cb() {
  // Register the exit-cb in reverse order (3,2,1)
  // so that they'd be called in 1,2,3 order.
  __libc_register_thread_exit_callback(&exit_cb_3);
  __libc_register_thread_exit_callback(&exit_cb_2);
  __libc_register_thread_exit_callback(&exit_cb_1);
}

int main() {
  test_register_thread_exit_cb();
  return 0;
}
#else
int main() {
  return 0;
}
#endif  // __BIONIC__
