/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#include <errno.h>
#include <sys/shm.h>

#include <android-base/file.h>
#include <gtest/gtest.h>

TEST(sys_shm, smoke) {
  if (shmctl(-1, IPC_STAT, nullptr) == -1 && errno == ENOSYS) {
    GTEST_SKIP() << "no <sys/shm.h> support in this kernel";
  }

  // Create a segment.
  TemporaryDir dir;
  key_t key = ftok(dir.path, 1);
  int id = shmget(key, 1234, IPC_CREAT|0666);
  ASSERT_NE(id, -1);

  // Check segment info.
  shmid_ds ds = {};
  ASSERT_EQ(0, shmctl(id, IPC_STAT, &ds));
  ASSERT_EQ(1234U, ds.shm_segsz);

  // Attach.
  void* p = shmat(id, nullptr, SHM_RDONLY);
  ASSERT_NE(p, nullptr);

  // Detach.
  ASSERT_EQ(0, shmdt(p));

  // Destroy the segment.
  ASSERT_EQ(0, shmctl(id, IPC_RMID, nullptr));
}

TEST(sys_shm, shmat_failure) {
  errno = 0;
  ASSERT_EQ(reinterpret_cast<void*>(-1), shmat(-1, nullptr, SHM_RDONLY));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
}

TEST(sys_shm, shmctl_failure) {
  errno = 0;
  ASSERT_EQ(-1, shmctl(-1, IPC_STAT, nullptr));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
}

TEST(sys_shm, shmdt_failure) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  errno = 0;
  ASSERT_EQ(-1, shmdt(nullptr));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
#pragma clang diagnostic pop
}

TEST(sys_shm, shmget_failure) {
  errno = 0;
  ASSERT_EQ(-1, shmget(-1, 1234, 0));
  ASSERT_TRUE(errno == ENOENT || errno == ENOSYS);
}
