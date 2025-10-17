/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
#include <gtest/gtest.h>

#include <errno.h>
#include <sys/sem.h>

#include <android-base/file.h>

#include "utils.h"

TEST(sys_sem, smoke) {
  if (semctl(-1, 0, IPC_RMID) == -1 && errno == ENOSYS) {
    GTEST_SKIP() << "no <sys/sem.h> support in this kernel";
  }

  // Create a semaphore.
  TemporaryDir dir;
  key_t key = ftok(dir.path, 1);
  int id = semget(key, 1, IPC_CREAT|0666);
  ASSERT_NE(id, -1);

  // Check semaphore info.
  semid_ds ds = {};
  ASSERT_EQ(0, semctl(id, 0, IPC_STAT, &ds));
  ASSERT_EQ(1U, ds.sem_nsems);

  ASSERT_EQ(0, semctl(id, 0, GETVAL));

  // Increment.
  sembuf ops[] = {{ .sem_num = 0, .sem_op = 1, .sem_flg = 0 }};
  ASSERT_EQ(0, semop(id, ops, 1));
  ASSERT_EQ(1, semctl(id, 0, GETVAL));

  // Test timeouts.
  timespec ts = { .tv_sec = 0, .tv_nsec = 100 };
  ops[0] = { .sem_num = 0, .sem_op = 0, .sem_flg = 0 };
  errno = 0;
  ASSERT_EQ(-1, semtimedop(id, ops, 1, &ts));
  ASSERT_ERRNO(EAGAIN);
  ASSERT_EQ(1, semctl(id, 0, GETVAL));

  // Decrement.
  ops[0] = { .sem_num = 0, .sem_op = -1, .sem_flg = 0 };
  ASSERT_EQ(0, semop(id, ops, 1));
  ASSERT_EQ(0, semctl(id, 0, GETVAL));

  // Destroy the semaphore.
  ASSERT_EQ(0, semctl(id, 0, IPC_RMID));
}

TEST(sys_sem, semget_failure) {
  errno = 0;
  ASSERT_EQ(-1, semget(-1, -1, 0));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
}

TEST(sys_sem, semctl_failure) {
  errno = 0;
  ASSERT_EQ(-1, semctl(-1, 0, IPC_RMID));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
}

TEST(sys_sem, semop_failure) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  errno = 0;
  ASSERT_EQ(-1, semop(-1, nullptr, 0));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
#pragma clang diagnostic pop
}

TEST(sys_sem, semtimedop_failure) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  errno = 0;
  ASSERT_EQ(-1, semtimedop(-1, nullptr, 0, nullptr));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
#pragma clang diagnostic pop
}

TEST(sys_sem, union_semun) {
  // https://github.com/android-ndk/ndk/issues/400
#if defined(__BIONIC__)
  semun arg;
  semid_ds i1;
  seminfo i2;
  unsigned short a[] = { 1u, 2u };
  arg.val = 123;
  arg.buf = &i1;
  arg.array = a;
  arg.__buf = &i2;
#else
  // glibc already mostly removed this cruft (although it's still in <linux/sem.h>).
#endif
}
