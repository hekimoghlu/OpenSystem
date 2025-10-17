/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#ifndef _UAPI_LINUX_UTSNAME_H
#define _UAPI_LINUX_UTSNAME_H
#define __OLD_UTS_LEN 8
struct oldold_utsname {
  char sysname[9];
  char nodename[9];
  char release[9];
  char version[9];
  char machine[9];
};
#define __NEW_UTS_LEN 64
struct old_utsname {
  char sysname[65];
  char nodename[65];
  char release[65];
  char version[65];
  char machine[65];
};
struct new_utsname {
  char sysname[__NEW_UTS_LEN + 1];
  char nodename[__NEW_UTS_LEN + 1];
  char release[__NEW_UTS_LEN + 1];
  char version[__NEW_UTS_LEN + 1];
  char machine[__NEW_UTS_LEN + 1];
  char domainname[__NEW_UTS_LEN + 1];
};
#endif
