/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
#include "server_setup.h"

/* Purpose
 *
 * Resolve the given name, using system name resolve functions (NOT any
 * function provided by libcurl). Used to see if the name exists and thus if
 * we can allow a test case to use it for testing.
 *
 * Like if 'localhost' actual exists etc.
 *
 */

#include <signal.h>
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef _XOPEN_SOURCE_EXTENDED
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
 */#include <arpa/inet.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

#define ENABLE_CURLX_PRINTF
/* make the curlx header define all printf() functions to use the curlx_*
   versions instead */
#include "curlx.h" /* from the private lib dir */
#include "util.h"

/* include memdebug.h last */
#include "memdebug.h"

static bool use_ipv6 = FALSE;
static const char *ipv_inuse = "IPv4";

const char *serverlogfile = ""; /* for a util.c function we don't use */

int main(int argc, char *argv[])
{
  int arg = 1;
  const char *host = NULL;
  int rc = 0;

  while(argc>arg) {
    if(!strcmp("--version", argv[arg])) {
      printf("resolve IPv4%s\n",
#if defined(CURLRES_IPV6)
             "/IPv6"
#else
             ""
#endif
             );
      return 0;
    }
    else if(!strcmp("--ipv6", argv[arg])) {
      ipv_inuse = "IPv6";
      use_ipv6 = TRUE;
      arg++;
    }
    else if(!strcmp("--ipv4", argv[arg])) {
      /* for completeness, we support this option as well */
      ipv_inuse = "IPv4";
      use_ipv6 = FALSE;
      arg++;
    }
    else {
      host = argv[arg++];
    }
  }
  if(!host) {
    puts("Usage: resolve [option] <host>\n"
         " --version\n"
         " --ipv4"
#if defined(CURLRES_IPV6)
         "\n --ipv6"
#endif
         );
    return 1;
  }

#ifdef _WIN32
  win32_init();
  atexit(win32_cleanup);
#endif

#if defined(CURLRES_IPV6)
  if(use_ipv6) {
    /* Check that the system has IPv6 enabled before checking the resolver */
    curl_socket_t s = socket(PF_INET6, SOCK_DGRAM, 0);
    if(s == CURL_SOCKET_BAD)
      /* an IPv6 address was requested and we can't get/use one */
      rc = -1;
    else {
      sclose(s);
    }
  }

  if(rc == 0) {
    /* getaddrinfo() resolve */
    struct addrinfo *ai;
    struct addrinfo hints;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = use_ipv6 ? PF_INET6 : PF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = 0;
    /* Use parenthesis around functions to stop them from being replaced by
       the macro in memdebug.h */
    rc = (getaddrinfo)(host, "80", &hints, &ai);
    if(rc == 0)
      (freeaddrinfo)(ai);
  }
#else
  if(use_ipv6) {
    puts("IPv6 support has been disabled in this program");
    return 1;
  }
  else {
    /* gethostbyname() resolve */
    struct hostent *he;

    he = gethostbyname(host);

    rc = !he;
  }
#endif

  if(rc)
    printf("Resolving %s '%s' didn't work\n", ipv_inuse, host);

  return !!rc;
}

