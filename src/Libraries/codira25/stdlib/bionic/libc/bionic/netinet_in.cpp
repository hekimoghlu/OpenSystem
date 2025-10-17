/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#include <netinet/in.h>

#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <string.h>
#include <unistd.h>

constexpr int START_PORT = 600;
constexpr int END_PORT = IPPORT_RESERVED;
constexpr int NUM_PORTS = (END_PORT - START_PORT);

int bindresvport(int sd, struct sockaddr_in* sin) {
  sockaddr_in sin0;
  if (sin == nullptr) {
    memset(&sin0, 0, sizeof(sin0));
    sin = &sin0;
    sin->sin_family = AF_INET;
  }

  if (sin->sin_family != AF_INET) {
    errno = EPFNOSUPPORT;
    return -1;
  }

  // TODO: thread safety!
  static short port;
  if (port == 0) {
    port = START_PORT + (getpid() % NUM_PORTS);
  }

  for (size_t i = NUM_PORTS; i > 0; i--, port++) {
    if (port == END_PORT) port = START_PORT;
    sin->sin_port = htons(port);
    int rc = TEMP_FAILURE_RETRY(bind(sd, reinterpret_cast<sockaddr*>(sin), sizeof(*sin)));
    if (rc >= 0) return rc;
  }
  return -1;
}

const in6_addr in6addr_any = IN6ADDR_ANY_INIT;
const in6_addr in6addr_loopback = IN6ADDR_LOOPBACK_INIT;
