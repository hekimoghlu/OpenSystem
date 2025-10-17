/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
#include "test.h"

#include <limits.h>
#include <assert.h>

#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

#define TEST_HANG_TIMEOUT 60 * 1000
#define MAX_EASY_HANDLES 3

static int counter[MAX_EASY_HANDLES];
static CURL *easy[MAX_EASY_HANDLES];
static curl_socket_t sockets[MAX_EASY_HANDLES];
static int res = 0;

static size_t callback(char *ptr, size_t size, size_t nmemb, void *data)
{
  ssize_t idx = ((CURL **) data) - easy;
  curl_socket_t sock;
  long longdata;
  CURLcode code;
  const size_t failure = (size && nmemb) ? 0 : 1;
  (void)ptr;

  counter[idx] += (int)(size * nmemb);

  /* Get socket being used for this easy handle, otherwise CURL_SOCKET_BAD */
  CURL_IGNORE_DEPRECATION(
    code = curl_easy_getinfo(easy[idx], CURLINFO_LASTSOCKET, &longdata);
  )
  if(CURLE_OK != code) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed, "
            "with code %d (%s)\n",
            __FILE__, __LINE__, (int)code, curl_easy_strerror(code));
    res = TEST_ERR_MAJOR_BAD;
    return failure;
  }
  if(longdata == -1L)
    sock = CURL_SOCKET_BAD;
  else
    sock = (curl_socket_t)longdata;

  if(sock != CURL_SOCKET_BAD) {
    /* Track relationship between this easy handle and the socket. */
    if(sockets[idx] == CURL_SOCKET_BAD) {
      /* An easy handle without previous socket, record the socket. */
      sockets[idx] = sock;
    }
    else if(sock != sockets[idx]) {
      /* An easy handle with a socket different to previously
         tracked one, log and fail right away. Known bug #37. */
      fprintf(stderr, "Handle %d started on socket %d and moved to %d\n",
              curlx_sztosi(idx), (int)sockets[idx], (int)sock);
      res = TEST_ERR_MAJOR_BAD;
      return failure;
    }
  }
  return size * nmemb;
}

enum HandleState {
  ReadyForNewHandle,
  NeedSocketForNewHandle,
  NoMoreHandles
};

int test(char *url)
{
  CURLM *multi = NULL;
  int running;
  int i;
  int num_handles = 0;
  enum HandleState state = ReadyForNewHandle;
  size_t urllen = strlen(url) + 4 + 1;
  char *full_url = malloc(urllen);

  start_test_timing();

  if(!full_url) {
    fprintf(stderr, "Not enough memory for full url\n");
    return TEST_ERR_MAJOR_BAD;
  }

  for(i = 0; i < MAX_EASY_HANDLES; ++i) {
    easy[i] = NULL;
    sockets[i] = CURL_SOCKET_BAD;
  }

  res_global_init(CURL_GLOBAL_ALL);
  if(res) {
    free(full_url);
    return res;
  }

  multi_init(multi);

  for(;;) {
    struct timeval interval;
    fd_set fdread;
    fd_set fdwrite;
    fd_set fdexcep;
    long timeout = -99;
    int maxfd = -99;
    bool found_new_socket = FALSE;

    /* Start a new handle if we aren't at the max */
    if(state == ReadyForNewHandle) {
      easy_init(easy[num_handles]);

      if(num_handles % 3 == 2) {
        msnprintf(full_url, urllen, "%s0200", url);
        easy_setopt(easy[num_handles], CURLOPT_HTTPAUTH, CURLAUTH_NTLM);
      }
      else {
        msnprintf(full_url, urllen, "%s0100", url);
        easy_setopt(easy[num_handles], CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
      }
      easy_setopt(easy[num_handles], CURLOPT_FRESH_CONNECT, 1L);
      easy_setopt(easy[num_handles], CURLOPT_URL, full_url);
      easy_setopt(easy[num_handles], CURLOPT_VERBOSE, 1L);
      easy_setopt(easy[num_handles], CURLOPT_HTTPGET, 1L);
      easy_setopt(easy[num_handles], CURLOPT_USERPWD, "testuser:testpass");
      easy_setopt(easy[num_handles], CURLOPT_WRITEFUNCTION, callback);
      easy_setopt(easy[num_handles], CURLOPT_WRITEDATA, easy + num_handles);
      easy_setopt(easy[num_handles], CURLOPT_HEADER, 1L);

      multi_add_handle(multi, easy[num_handles]);
      num_handles += 1;
      state = NeedSocketForNewHandle;
    }

    multi_perform(multi, &running);

    fprintf(stderr, "%s:%d running %d state %d\n",
            __FILE__, __LINE__, running, state);

    abort_on_test_timeout();

    if(!running && state == NoMoreHandles)
      break; /* done */

    FD_ZERO(&fdread);
    FD_ZERO(&fdwrite);
    FD_ZERO(&fdexcep);

    multi_fdset(multi, &fdread, &fdwrite, &fdexcep, &maxfd);

    /* At this point, maxfd is guaranteed to be greater or equal than -1. */

    if(state == NeedSocketForNewHandle) {
      if(maxfd != -1 && !found_new_socket) {
        fprintf(stderr, "Warning: socket did not open immediately for new "
                "handle (trying again)\n");
        continue;
      }
      state = num_handles < MAX_EASY_HANDLES ? ReadyForNewHandle
                                             : NoMoreHandles;
      fprintf(stderr, "%s:%d new state %d\n",
              __FILE__, __LINE__, state);
    }

    multi_timeout(multi, &timeout);

    /* At this point, timeout is guaranteed to be greater or equal than -1. */

    fprintf(stderr, "%s:%d num_handles %d timeout %ld running %d\n",
            __FILE__, __LINE__, num_handles, timeout, running);

    if(timeout != -1L) {
      int itimeout;
#if LONG_MAX > INT_MAX
      itimeout = (timeout > (long)INT_MAX) ? INT_MAX : (int)timeout;
#else
      itimeout = (int)timeout;
#endif
      interval.tv_sec = itimeout/1000;
      interval.tv_usec = (itimeout%1000)*1000;
    }
    else {
      interval.tv_sec = 0;
      interval.tv_usec = 5000;

      /* if there's no timeout and we get here on the last handle, we may
         already have read the last part of the stream so waiting makes no
         sense */
      if(!running && num_handles == MAX_EASY_HANDLES) {
        break;
      }
    }

    select_test(maxfd + 1, &fdread, &fdwrite, &fdexcep, &interval);

    abort_on_test_timeout();
  }

test_cleanup:

  /* proper cleanup sequence - type PB */

  for(i = 0; i < MAX_EASY_HANDLES; i++) {
    printf("Data connection %d: %d\n", i, counter[i]);
    curl_multi_remove_handle(multi, easy[i]);
    curl_easy_cleanup(easy[i]);
  }

  curl_multi_cleanup(multi);
  curl_global_cleanup();

  free(full_url);

  return res;
}
