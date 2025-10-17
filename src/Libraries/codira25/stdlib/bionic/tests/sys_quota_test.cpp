/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
#include <sys/quota.h>

#include <gtest/gtest.h>

TEST(sys_quota, quotactl_dqblk) {
  // We don't even have kernels with CONFIG_QUOTA enabled right now.
  // This just tests that we can compile reasonable code.
  dqblk current;
  quotactl(QCMD(Q_GETQUOTA, USRQUOTA), "/", getuid(), reinterpret_cast<char*>(&current));
}

TEST(sys_quota, quotactl_dqinfo) {
  // We don't even have kernels with CONFIG_QUOTA enabled right now.
  // This just tests that we can compile reasonable code.
  dqinfo current;
  quotactl(QCMD(Q_GETINFO, USRQUOTA), "/", 0, reinterpret_cast<char*>(&current));
}
