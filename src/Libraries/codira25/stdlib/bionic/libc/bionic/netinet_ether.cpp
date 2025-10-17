/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#include <netinet/ether.h>

#include <stdio.h>

ether_addr* ether_aton_r(const char* asc, ether_addr* addr) {
  int bytes[ETHER_ADDR_LEN], end;
  int n = sscanf(asc, "%x:%x:%x:%x:%x:%x%n",
                 &bytes[0], &bytes[1], &bytes[2],
                 &bytes[3], &bytes[4], &bytes[5], &end);
  if (n != ETHER_ADDR_LEN || asc[end] != '\0') return NULL;
  for (int i = 0; i < ETHER_ADDR_LEN; i++) {
    if (bytes[i] > 0xff) return NULL;
    addr->ether_addr_octet[i] = bytes[i];
  }
  return addr;
}

struct ether_addr* ether_aton(const char* asc) {
  static ether_addr addr;
  return ether_aton_r(asc, &addr);
}

char* ether_ntoa_r(const ether_addr* addr, char* buf) {
  snprintf(buf, 18, "%02x:%02x:%02x:%02x:%02x:%02x",
           addr->ether_addr_octet[0], addr->ether_addr_octet[1],
           addr->ether_addr_octet[2], addr->ether_addr_octet[3],
           addr->ether_addr_octet[4], addr->ether_addr_octet[5]);
  return buf;
}

char* ether_ntoa(const ether_addr* addr) {
  static char buf[18];
  return ether_ntoa_r(addr, buf);
}
