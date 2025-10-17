/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
/* <DESC>
 * Demonstrate curl_easy_send() and curl_easy_recv() usage.
 * </DESC>
 */

#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

/* Auxiliary function that waits on the socket. */
static int wait_on_socket(curl_socket_t sockfd, int for_recv, long timeout_ms)
{
  struct timeval tv;
  fd_set infd, outfd, errfd;
  int res;

  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (int)(timeout_ms % 1000) * 1000;

  FD_ZERO(&infd);
  FD_ZERO(&outfd);
  FD_ZERO(&errfd);

/* Avoid this warning with pre-2020 Cygwin/MSYS releases:
 * warning: conversion to 'long unsigned int' from 'curl_socket_t' {aka 'int'}
 * may change the sign of the result [-Wsign-conversion]
 */
#if defined(__GNUC__) && defined(__CYGWIN__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
  FD_SET(sockfd, &errfd); /* always check for error */

  if(for_recv) {
    FD_SET(sockfd, &infd);
  }
  else {
    FD_SET(sockfd, &outfd);
  }
#if defined(__GNUC__) && defined(__CYGWIN__)
#pragma GCC diagnostic pop
#endif

  /* select() returns the number of signalled sockets or -1 */
  res = select((int)sockfd + 1, &infd, &outfd, &errfd, &tv);
  return res;
}

int main(void)
{
  CURL *curl;
  /* Minimalistic http request */
  const char *request = "GET / HTTP/1.0\r\nHost: example.com\r\n\r\n";
  size_t request_len = strlen(request);

  /* A general note of caution here: if you are using curl_easy_recv() or
     curl_easy_send() to implement HTTP or _any_ other protocol libcurl
     supports "natively", you are doing it wrong and you should stop.

     This example uses HTTP only to show how to use this API, it does not
     suggest that writing an application doing this is sensible.
  */

  curl = curl_easy_init();
  if(curl) {
    CURLcode res;
    curl_socket_t sockfd;
    size_t nsent_total = 0;

    curl_easy_setopt(curl, CURLOPT_URL, "https://example.com");
    /* Do not do the transfer - only connect to host */
    curl_easy_setopt(curl, CURLOPT_CONNECT_ONLY, 1L);
    res = curl_easy_perform(curl);

    if(res != CURLE_OK) {
      printf("Error: %s\n", curl_easy_strerror(res));
      return 1;
    }

    /* Extract the socket from the curl handle - we need it for waiting. */
    res = curl_easy_getinfo(curl, CURLINFO_ACTIVESOCKET, &sockfd);

    if(res != CURLE_OK) {
      printf("Error: %s\n", curl_easy_strerror(res));
      return 1;
    }

    printf("Sending request.\n");

    do {
      /* Warning: This example program may loop indefinitely.
       * A production-quality program must define a timeout and exit this loop
       * as soon as the timeout has expired. */
      size_t nsent;
      do {
        nsent = 0;
        res = curl_easy_send(curl, request + nsent_total,
            request_len - nsent_total, &nsent);
        nsent_total += nsent;

        if(res == CURLE_AGAIN && !wait_on_socket(sockfd, 0, 60000L)) {
          printf("Error: timeout.\n");
          return 1;
        }
      } while(res == CURLE_AGAIN);

      if(res != CURLE_OK) {
        printf("Error: %s\n", curl_easy_strerror(res));
        return 1;
      }

      printf("Sent %lu bytes.\n", (unsigned long)nsent);

    } while(nsent_total < request_len);

    printf("Reading response.\n");

    for(;;) {
      /* Warning: This example program may loop indefinitely (see above). */
      char buf[1024];
      size_t nread;
      do {
        nread = 0;
        res = curl_easy_recv(curl, buf, sizeof(buf), &nread);

        if(res == CURLE_AGAIN && !wait_on_socket(sockfd, 1, 60000L)) {
          printf("Error: timeout.\n");
          return 1;
        }
      } while(res == CURLE_AGAIN);

      if(res != CURLE_OK) {
        printf("Error: %s\n", curl_easy_strerror(res));
        break;
      }

      if(nread == 0) {
        /* end of the response */
        break;
      }

      printf("Received %lu bytes.\n", (unsigned long)nread);
    }

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return 0;
}
