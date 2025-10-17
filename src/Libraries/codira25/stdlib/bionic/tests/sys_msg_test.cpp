/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#include <sys/msg.h>

#include <android-base/file.h>

TEST(sys_msg, smoke) {
  if (msgctl(-1, IPC_STAT, nullptr) == -1 && errno == ENOSYS) {
    GTEST_SKIP() << "no <sys/msg.h> support in this kernel";
  }

  // Create a queue.
  TemporaryDir dir;
  key_t key = ftok(dir.path, 1);
  int id = msgget(key, IPC_CREAT|0666);
  ASSERT_NE(id, -1);

  // Queue should be empty.
  msqid_ds ds = {};
  ASSERT_EQ(0, msgctl(id, IPC_STAT, &ds));
  ASSERT_EQ(0U, ds.msg_qnum);
  ASSERT_EQ(0U, ds.msg_cbytes);

  // Send a message.
  struct {
    long type;
    char data[32];
  } msg = { 1, "hello world" };
  ASSERT_EQ(0, msgsnd(id, &msg, sizeof(msg.data), 0));

  // Queue should be non-empty.
  ASSERT_EQ(0, msgctl(id, IPC_STAT, &ds));
  ASSERT_EQ(1U, ds.msg_qnum);
  ASSERT_EQ(sizeof(msg.data), ds.msg_cbytes);

  // Read the message.
  msg = {};
  ASSERT_EQ(static_cast<ssize_t>(sizeof(msg.data)),
            msgrcv(id, &msg, sizeof(msg.data), 0, 0));
  ASSERT_EQ(1, msg.type);
  ASSERT_STREQ("hello world", msg.data);

  // Destroy the queue.
  ASSERT_EQ(0, msgctl(id, IPC_RMID, nullptr));
}

TEST(sys_msg, msgctl_failure) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  errno = 0;
  ASSERT_EQ(-1, msgctl(-1, IPC_STAT, nullptr));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
#pragma clang diagnostic pop
}

TEST(sys_msg, msgget_failure) {
  errno = 0;
  ASSERT_EQ(-1, msgget(-1, 0));
  ASSERT_TRUE(errno == ENOENT || errno == ENOSYS);
}

TEST(sys_msg, msgrcv_failure) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  errno = 0;
  ASSERT_EQ(-1, msgrcv(-1, nullptr, 0, 0, 0));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
#pragma clang diagnostic pop
}

TEST(sys_msg, msgsnd_failure) {
  struct {
    long type;
    char data[1];
  } msg = { 1, "" };
  errno = 0;
  ASSERT_EQ(-1, msgsnd(-1, &msg, sizeof(msg.data), 0));
  ASSERT_TRUE(errno == EINVAL || errno == ENOSYS);
}
