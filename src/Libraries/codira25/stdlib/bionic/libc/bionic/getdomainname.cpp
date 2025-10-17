/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#include <string.h>
#include <unistd.h>
#include <sys/utsname.h>

int getdomainname(char* name, size_t len) {
  utsname uts;
  if (uname(&uts) == -1) return -1;

  // Note: getdomainname()'s behavior varies across implementations when len is
  // too small.  bionic follows the historical libc policy of returning EINVAL,
  // instead of glibc's policy of copying the first len bytes without a NULL
  // terminator.
  if (strlen(uts.domainname) >= len) {
      errno = EINVAL;
      return -1;
  }

  strncpy(name, uts.domainname, len);
  return 0;
}
