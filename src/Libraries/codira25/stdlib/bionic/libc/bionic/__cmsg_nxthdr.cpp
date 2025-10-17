/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#include <sys/socket.h>

cmsghdr* __cmsg_nxthdr(msghdr* msg, cmsghdr* cmsg) {
  cmsghdr* ptr;
  ptr = reinterpret_cast<cmsghdr*>(reinterpret_cast<char*>(cmsg) + CMSG_ALIGN(cmsg->cmsg_len));
  size_t len = reinterpret_cast<char*>(ptr+1) - reinterpret_cast<char*>(msg->msg_control);
  if (len > msg->msg_controllen) {
    return nullptr;
  }
  return ptr;
}
