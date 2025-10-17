/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#include <net/if.h>

#include <errno.h>
#include <ifaddrs.h>
#include <linux/if_packet.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/sockios.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include "private/ScopedFd.h"

#include "bionic_netlink.h"

char* if_indextoname(unsigned ifindex, char* ifname) {
  ScopedFd s(socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC, 0));
  if (s.get() == -1) return nullptr;

  ifreq ifr = {.ifr_ifindex = static_cast<int>(ifindex)};
  return (ioctl(s.get(), SIOCGIFNAME, &ifr) == -1) ? nullptr
                                                   : strncpy(ifname, ifr.ifr_name, IFNAMSIZ);
}

unsigned if_nametoindex(const char* ifname) {
  ScopedFd s(socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC, 0));
  if (s.get() == -1) return 0;

  ifreq ifr = {};
  strncpy(ifr.ifr_name, ifname, sizeof(ifr.ifr_name));
  ifr.ifr_name[IFNAMSIZ - 1] = 0;
  return (ioctl(s.get(), SIOCGIFINDEX, &ifr) == -1) ? 0 : ifr.ifr_ifindex;
}

struct if_list {
  if_list* next;
  struct if_nameindex data;

  explicit if_list(if_list** list) {
    // push_front onto `list`.
    next = *list;
    *list = this;
  }

  static void Free(if_list* list, bool names_too) {
    while (list) {
      if_list* it = list;
      list = it->next;
      if (names_too) free(it->data.if_name);
      free(it);
    }
  }
};

static void __if_nameindex_callback(void* context, nlmsghdr* hdr) {
  if_list** list = reinterpret_cast<if_list**>(context);
  if (hdr->nlmsg_type == RTM_NEWLINK) {
    ifinfomsg* ifi = reinterpret_cast<ifinfomsg*>(NLMSG_DATA(hdr));

    // Create a new entry and set the interface index.
    if_list* new_link = new if_list(list);
    new_link->data.if_index = ifi->ifi_index;

    // Go through the various bits of information and find the name.
    rtattr* rta = IFLA_RTA(ifi);
    size_t rta_len = IFLA_PAYLOAD(hdr);
    while (RTA_OK(rta, rta_len)) {
      if (rta->rta_type == IFLA_IFNAME) {
        new_link->data.if_name = strndup(reinterpret_cast<char*>(RTA_DATA(rta)), RTA_PAYLOAD(rta));
      }
      rta = RTA_NEXT(rta, rta_len);
    }
  }
}

struct if_nameindex* if_nameindex() {
  if_list* list = nullptr;

  // Open the netlink socket and ask for all the links;
  NetlinkConnection nc;
  bool okay = nc.SendRequest(RTM_GETLINK) && nc.ReadResponses(__if_nameindex_callback, &list);
  if (!okay) {
    if_list::Free(list, true);
    return nullptr;
  }

  // Count the interfaces.
  size_t interface_count = 0;
  for (if_list* it = list; it != nullptr; it = it->next) {
    ++interface_count;
  }

  // Build the array POSIX requires us to return.
  struct if_nameindex* result = new struct if_nameindex[interface_count + 1];
  if (result) {
    struct if_nameindex* out = result;
    for (if_list* it = list; it != nullptr; it = it->next) {
      out->if_index = it->data.if_index;
      out->if_name = it->data.if_name;
      ++out;
    }
    out->if_index = 0;
    out->if_name = nullptr;
  }

  // Free temporary storage.
  if_list::Free(list, false);

  return result;
}

void if_freenameindex(struct if_nameindex* array) {
  if (array == nullptr) return;

  struct if_nameindex* ptr = array;
  while (ptr->if_index != 0 || ptr->if_name != nullptr) {
    free(ptr->if_name);
    ++ptr;
  }

  delete[] array;
}
