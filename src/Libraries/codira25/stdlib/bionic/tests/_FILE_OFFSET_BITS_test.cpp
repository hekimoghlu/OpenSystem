/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#define _FILE_OFFSET_BITS 64

#include <gtest/gtest.h>

#include <fcntl.h>

TEST(fcntl, f_getlk_FOB64) {
  int fd = open("/proc/version", O_RDONLY);
  ASSERT_TRUE(fd != -1);

  struct flock check_lock;
  check_lock.l_type = F_WRLCK;
  check_lock.l_start = 0;
  check_lock.l_whence = SEEK_SET;
  check_lock.l_len = 0;

  ASSERT_EQ(0, fcntl(fd, F_GETLK, &check_lock));
  close(fd);
}
