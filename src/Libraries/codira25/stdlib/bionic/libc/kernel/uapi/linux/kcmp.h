/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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
#ifndef _UAPI_LINUX_KCMP_H
#define _UAPI_LINUX_KCMP_H
#include <linux/types.h>
enum kcmp_type {
  KCMP_FILE,
  KCMP_VM,
  KCMP_FILES,
  KCMP_FS,
  KCMP_SIGHAND,
  KCMP_IO,
  KCMP_SYSVSEM,
  KCMP_EPOLL_TFD,
  KCMP_TYPES,
};
struct kcmp_epoll_slot {
  __u32 efd;
  __u32 tfd;
  __u32 toff;
};
#endif
