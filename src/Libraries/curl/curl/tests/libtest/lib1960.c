/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#ifdef HAVE_INET_PTON

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif

#include "memdebug.h"

/* to prevent libcurl from closing our socket */
static int closesocket_cb(void *clientp, curl_socket_t item)
{
  (void)clientp;
  (void)item;
  return 0;
}

/* provide our own socket */
static curl_socket_t socket_cb(void *clientp,
                               curlsocktype purpose,
                               struct curl_sockaddr *address)
{
  int s = *(int *)clientp;
  (void)purpose;
  (void)address;
  return (curl_socket_t)s;
}

/* tell libcurl the socket is connected */
static int sockopt_cb(void *clientp,
                      curl_socket_t curlfd,
                      curlsocktype purpose)
{
  (void)clientp;
  (void)curlfd;
  (void)purpose;
  return CURL_SOCKOPT_ALREADY_CONNECTED;
}

/* Expected args: URL IP PORT */
int test(char *URL)
{
  CURL *curl = NULL;
  CURLcode res = TEST_ERR_MAJOR_BAD;
  int status;
  curl_socket_t client_fd = CURL_SOCKET_BAD;
  struct sockaddr_in serv_addr;
  unsigned short port;

  if(!strcmp("check", URL))
    return 0; /* no output makes it not skipped */

  port = (unsigned short)atoi(libtest_arg3);

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  /*
   * This code connects to the TCP port "manually" so that we then can hand
   * over this socket as "already connected" to libcurl and make sure that
   * this works.
   */
  client_fd = socket(AF_INET, SOCK_STREAM, 0);
  if(client_fd == CURL_SOCKET_BAD) {
    fprintf(stderr, "socket creation error\n");
    goto test_cleanup;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  if(inet_pton(AF_INET, libtest_arg2, &serv_addr.sin_addr) <= 0) {
    fprintf(stderr, "inet_pton failed\n");
    goto test_cleanup;
  }

  status = connect(client_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
  if(status < 0) {
    fprintf(stderr, "connection failed\n");
    goto test_cleanup;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    goto test_cleanup;
  }

  test_setopt(curl, CURLOPT_VERBOSE, 1L);
  test_setopt(curl, CURLOPT_OPENSOCKETFUNCTION, socket_cb);
  test_setopt(curl, CURLOPT_OPENSOCKETDATA, &client_fd);
  test_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_cb);
  test_setopt(curl, CURLOPT_SOCKOPTDATA, NULL);
  test_setopt(curl, CURLOPT_CLOSESOCKETFUNCTION, closesocket_cb);
  test_setopt(curl, CURLOPT_CLOSESOCKETDATA, NULL);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);
  test_setopt(curl, CURLOPT_HEADER, 1L);
  test_setopt(curl, CURLOPT_URL, URL);

  res = curl_easy_perform(curl);

test_cleanup:
  if(client_fd != CURL_SOCKET_BAD)
    sclose(client_fd);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
#else
int test(char *URL)
{
  (void)URL;
  printf("lacks inet_pton\n");
  return 0;
}
#endif
