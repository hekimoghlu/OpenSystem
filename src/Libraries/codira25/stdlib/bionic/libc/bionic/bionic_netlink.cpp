/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#include "bionic_netlink.h"

#include <errno.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

#include "private/ErrnoRestorer.h"

NetlinkConnection::NetlinkConnection() {
  // The kernel keeps packets under 8KiB (NLMSG_GOODSIZE),
  // but that's a bit too large to go on the stack.
  size_ = 8192;
  data_ = new char[size_];
}

NetlinkConnection::~NetlinkConnection() {
  delete[] data_;
}

bool NetlinkConnection::SendRequest(int type) {
  // Rather than force all callers to check for the unlikely event of being
  // unable to allocate 8KiB, check here.
  if (data_ == nullptr) return false;

  // Did we open a netlink socket yet?
  if (fd_.get() == -1) {
    fd_.reset(socket(PF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_ROUTE));
    if (fd_.get() == -1) return false;
  }

  // Construct and send the message.
  struct NetlinkMessage {
    nlmsghdr hdr;
    rtgenmsg msg;
  } request;
  memset(&request, 0, sizeof(request));
  request.hdr.nlmsg_flags = NLM_F_DUMP | NLM_F_REQUEST;
  request.hdr.nlmsg_type = type;
  request.hdr.nlmsg_len = sizeof(request);
  request.msg.rtgen_family = AF_UNSPEC; // All families.
  return (TEMP_FAILURE_RETRY(send(fd_.get(), &request, sizeof(request), 0)) == sizeof(request));
}

bool NetlinkConnection::ReadResponses(void callback(void*, nlmsghdr*), void* context) {
  // Read through all the responses, handing interesting ones to the callback.
  ssize_t bytes_read;
  while ((bytes_read = TEMP_FAILURE_RETRY(recv(fd_.get(), data_, size_, 0))) > 0) {
    nlmsghdr* hdr = reinterpret_cast<nlmsghdr*>(data_);
    for (; NLMSG_OK(hdr, static_cast<size_t>(bytes_read)); hdr = NLMSG_NEXT(hdr, bytes_read)) {
      if (hdr->nlmsg_type == NLMSG_DONE) return true;
      if (hdr->nlmsg_type == NLMSG_ERROR) {
        nlmsgerr* err = reinterpret_cast<nlmsgerr*>(NLMSG_DATA(hdr));
        errno = (hdr->nlmsg_len >= NLMSG_LENGTH(sizeof(nlmsgerr))) ? -err->error : EIO;
        return false;
      }
      callback(context, hdr);
    }
  }

  // We only get here if recv fails before we see a NLMSG_DONE.
  return false;
}
