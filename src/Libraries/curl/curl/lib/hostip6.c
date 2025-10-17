/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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
#include "curl_setup.h"

/***********************************************************************
 * Only for IPv6-enabled builds
 **********************************************************************/
#ifdef CURLRES_IPV6

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef __VMS
#include <in.h>
#include <inet.h>
#endif

#include "urldata.h"
#include "sendf.h"
#include "hostip.h"
#include "hash.h"
#include "share.h"
#include "url.h"
#include "inet_pton.h"
#include "connect.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

/*
 * Curl_ipvalid() checks what CURL_IPRESOLVE_* requirements that might've
 * been set and returns TRUE if they are OK.
 */
bool Curl_ipvalid(struct Curl_easy *data, struct connectdata *conn)
{
  if(conn->ip_version == CURL_IPRESOLVE_V6)
    return Curl_ipv6works(data);

  return TRUE;
}

#if defined(CURLRES_SYNCH)

#ifdef DEBUG_ADDRINFO
static void dump_addrinfo(const struct Curl_addrinfo *ai)
{
  printf("dump_addrinfo:\n");
  for(; ai; ai = ai->ai_next) {
    char buf[INET6_ADDRSTRLEN];
    printf("    fam %2d, CNAME %s, ",
           ai->ai_family, ai->ai_canonname ? ai->ai_canonname : "<none>");
    Curl_printable_address(ai, buf, sizeof(buf));
    printf("%s\n", buf);
  }
}
#else
#define dump_addrinfo(x) Curl_nop_stmt
#endif

/*
 * Curl_getaddrinfo() when built IPv6-enabled (non-threading and
 * non-ares version).
 *
 * Returns name information about the given hostname and port number. If
 * successful, the 'addrinfo' is returned and the fourth argument will point
 * to memory we need to free after use. That memory *MUST* be freed with
 * Curl_freeaddrinfo(), nothing else.
 */
struct Curl_addrinfo *Curl_getaddrinfo(struct Curl_easy *data,
                                       const char *hostname,
                                       int port,
                                       int *waitp)
{
  struct addrinfo hints;
  struct Curl_addrinfo *res;
  int error;
  char sbuf[12];
  char *sbufptr = NULL;
#ifndef USE_RESOLVE_ON_IPS
  char addrbuf[128];
#endif
  int pf = PF_INET;

  *waitp = 0; /* synchronous response only */

  if((data->conn->ip_version != CURL_IPRESOLVE_V4) && Curl_ipv6works(data))
    /* The stack seems to be IPv6-enabled */
    pf = PF_UNSPEC;

  memset(&hints, 0, sizeof(hints));
  hints.ai_family = pf;
  hints.ai_socktype = (data->conn->transport == TRNSPRT_TCP) ?
    SOCK_STREAM : SOCK_DGRAM;

#ifndef USE_RESOLVE_ON_IPS
  /*
   * The AI_NUMERICHOST must not be set to get synthesized IPv6 address from
   * an IPv4 address on iOS and Mac OS X.
   */
  if((1 == Curl_inet_pton(AF_INET, hostname, addrbuf)) ||
     (1 == Curl_inet_pton(AF_INET6, hostname, addrbuf))) {
    /* the given address is numerical only, prevent a reverse lookup */
    hints.ai_flags = AI_NUMERICHOST;
  }
#endif

  if(port) {
    msnprintf(sbuf, sizeof(sbuf), "%d", port);
    sbufptr = sbuf;
  }

  error = Curl_getaddrinfo_ex(hostname, sbufptr, &hints, &res);
  if(error) {
    infof(data, "getaddrinfo(3) failed for %s:%d", hostname, port);
    return NULL;
  }

  if(port) {
    Curl_addrinfo_set_port(res, port);
  }

  dump_addrinfo(res);

  return res;
}
#endif /* CURLRES_SYNCH */

#endif /* CURLRES_IPV6 */
