/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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

#ifdef __AMIGA__

#include <curl/curl.h>

#include "hostip.h"
#include "amigaos.h"

#ifdef HAVE_PROTO_BSDSOCKET_H
#  if defined(__amigaos4__)
#    include <bsdsocket/socketbasetags.h>
#  elif !defined(USE_AMISSL)
#    include <amitcp/socketbasetags.h>
#  endif
#  ifdef __libnix__
#    include <stabs.h>
#  endif
#endif

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

#ifdef HAVE_PROTO_BSDSOCKET_H

#ifdef __amigaos4__
/*
 * AmigaOS 4.x specific code
 */

/*
 * hostip4.c - Curl_ipv4_resolve_r() replacement code
 *
 * Logic that needs to be considered are the following build cases:
 * - newlib networking
 * - clib2 networking
 * - direct bsdsocket.library networking (usually AmiSSL builds)
 * Each with the threaded resolver enabled or not.
 *
 * With the threaded resolver enabled, try to use gethostbyname_r() where
 * available, otherwise (re)open bsdsocket.library and fallback to
 * gethostbyname().
 */

#include <proto/bsdsocket.h>

static struct SocketIFace *__CurlISocket = NULL;
static uint32 SocketFeatures = 0;

#define HAVE_BSDSOCKET_GETHOSTBYNAME_R 0x01
#define HAVE_BSDSOCKET_GETADDRINFO     0x02

CURLcode Curl_amiga_init(void)
{
  struct SocketIFace *ISocket;
  struct Library *base = OpenLibrary("bsdsocket.library", 4);

  if(base) {
    ISocket = (struct SocketIFace *)GetInterface(base, "main", 1, NULL);
    if(ISocket) {
      ULONG enabled = 0;

      SocketBaseTags(SBTM_SETVAL(SBTC_CAN_SHARE_LIBRARY_BASES), TRUE,
                     SBTM_GETREF(SBTC_HAVE_GETHOSTADDR_R_API), (ULONG)&enabled,
                     TAG_DONE);

      if(enabled) {
        SocketFeatures |= HAVE_BSDSOCKET_GETHOSTBYNAME_R;
      }

      __CurlISocket = ISocket;

      atexit(Curl_amiga_cleanup);

      return CURLE_OK;
    }
    CloseLibrary(base);
  }

  return CURLE_FAILED_INIT;
}

void Curl_amiga_cleanup(void)
{
  if(__CurlISocket) {
    struct Library *base = __CurlISocket->Data.LibBase;
    DropInterface((struct Interface *)__CurlISocket);
    CloseLibrary(base);
    __CurlISocket = NULL;
  }
}

#ifdef CURLRES_AMIGA
/*
 * Because we need to handle the different cases in hostip4.c at run-time,
 * not at compile-time, based on what was detected in Curl_amiga_init(),
 * we replace it completely with our own as to not complicate the baseline
 * code. Assumes malloc/calloc/free are thread safe because Curl_he2ai()
 * allocates memory also.
 */

struct Curl_addrinfo *Curl_ipv4_resolve_r(const char *hostname,
                                          int port)
{
  struct Curl_addrinfo *ai = NULL;
  struct hostent *h;
  struct SocketIFace *ISocket = __CurlISocket;

  if(SocketFeatures & HAVE_BSDSOCKET_GETHOSTBYNAME_R) {
    LONG h_errnop = 0;
    struct hostent *buf;

    buf = calloc(1, CURL_HOSTENT_SIZE);
    if(buf) {
      h = gethostbyname_r((STRPTR)hostname, buf,
                          (char *)buf + sizeof(struct hostent),
                          CURL_HOSTENT_SIZE - sizeof(struct hostent),
                          &h_errnop);
      if(h) {
        ai = Curl_he2ai(h, port);
      }
      free(buf);
    }
  }
  else {
    #ifdef CURLRES_THREADED
    /* gethostbyname() is not thread safe, so we need to reopen bsdsocket
     * on the thread's context
     */
    struct Library *base = OpenLibrary("bsdsocket.library", 4);
    if(base) {
      ISocket = (struct SocketIFace *)GetInterface(base, "main", 1, NULL);
      if(ISocket) {
        h = gethostbyname((STRPTR)hostname);
        if(h) {
          ai = Curl_he2ai(h, port);
        }
        DropInterface((struct Interface *)ISocket);
      }
      CloseLibrary(base);
    }
    #else
    /* not using threaded resolver - safe to use this as-is */
    h = gethostbyname(hostname);
    if(h) {
      ai = Curl_he2ai(h, port);
    }
    #endif
  }

  return ai;
}
#endif /* CURLRES_AMIGA */

#ifdef USE_AMISSL
#include <signal.h>
int Curl_amiga_select(int nfds, fd_set *readfds, fd_set *writefds,
                      fd_set *errorfds, struct timeval *timeout)
{
  int r = WaitSelect(nfds, readfds, writefds, errorfds, timeout, 0);
  /* Ensure Ctrl-C signal is actioned */
  if((r == -1) && (SOCKERRNO == EINTR))
    raise(SIGINT);
  return r;
}
#endif /* USE_AMISSL */

#elif !defined(USE_AMISSL) /* __amigaos4__ */
/*
 * Amiga OS3 specific code
 */

struct Library *SocketBase = NULL;
extern int errno, h_errno;

#ifdef __libnix__
void __request(const char *msg);
#else
# define __request(msg)       Printf(msg "\n\a")
#endif

void Curl_amiga_cleanup(void)
{
  if(SocketBase) {
    CloseLibrary(SocketBase);
    SocketBase = NULL;
  }
}

CURLcode Curl_amiga_init(void)
{
  if(!SocketBase)
    SocketBase = OpenLibrary("bsdsocket.library", 4);

  if(!SocketBase) {
    __request("No TCP/IP Stack running!");
    return CURLE_FAILED_INIT;
  }

  if(SocketBaseTags(SBTM_SETVAL(SBTC_ERRNOPTR(sizeof(errno))), (ULONG) &errno,
                    SBTM_SETVAL(SBTC_LOGTAGPTR), (ULONG) "curl",
                    TAG_DONE)) {
    __request("SocketBaseTags ERROR");
    return CURLE_FAILED_INIT;
  }

#ifndef __libnix__
  atexit(Curl_amiga_cleanup);
#endif

  return CURLE_OK;
}

#ifdef __libnix__
ADD2EXIT(Curl_amiga_cleanup, -50);
#endif

#endif /* !USE_AMISSL */

#endif /* HAVE_PROTO_BSDSOCKET_H */

#endif /* __AMIGA__ */

