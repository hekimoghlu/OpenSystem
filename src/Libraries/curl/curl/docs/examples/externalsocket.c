/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
 * Pass in a custom socket for libcurl to use.
 * </DESC>
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <curl/curl.h>

#ifdef _WIN32
#include <winsock2.h>
#define close closesocket
#else
#include <sys/types.h>        /*  socket types              */
#include <sys/socket.h>       /*  socket definitions        */
#include <netinet/in.h>
#include <arpa/inet.h>        /*  inet (3) functions         */
#include <unistd.h>           /*  misc. Unix functions      */
#endif

#include <errno.h>

/* The IP address and port number to connect to */
#define IPADDR "127.0.0.1"
#define PORTNUM 80

#ifndef INADDR_NONE
#define INADDR_NONE 0xffffffff
#endif

static size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
  size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
  return written;
}

static int closecb(void *clientp, curl_socket_t item)
{
  (void)clientp;
  printf("libcurl wants to close %d now\n", (int)item);
  return 0;
}

static curl_socket_t opensocket(void *clientp,
                                curlsocktype purpose,
                                struct curl_sockaddr *address)
{
  curl_socket_t sockfd;
  (void)purpose;
  (void)address;
  sockfd = *(curl_socket_t *)clientp;
  /* the actual externally set socket is passed in via the OPENSOCKETDATA
     option */
  return sockfd;
}

static int sockopt_callback(void *clientp, curl_socket_t curlfd,
                            curlsocktype purpose)
{
  (void)clientp;
  (void)curlfd;
  (void)purpose;
  /* This return code was added in libcurl 7.21.5 */
  return CURL_SOCKOPT_ALREADY_CONNECTED;
}

int main(void)
{
  CURL *curl;
  CURLcode res;
  struct sockaddr_in servaddr;  /*  socket address structure  */
  curl_socket_t sockfd;

#ifdef _WIN32
  WSADATA wsaData;
  int initwsa = WSAStartup(MAKEWORD(2, 2), &wsaData);
  if(initwsa) {
    printf("WSAStartup failed: %d\n", initwsa);
    return 1;
  }
#endif

  curl = curl_easy_init();
  if(curl) {
    /*
     * Note that libcurl internally thinks that you connect to the host and
     * port that you specify in the URL option.
     */
    curl_easy_setopt(curl, CURLOPT_URL, "http://99.99.99.99:9999");

    /* Create the socket "manually" */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd == CURL_SOCKET_BAD) {
      printf("Error creating listening socket.\n");
      return 3;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port   = htons(PORTNUM);

    servaddr.sin_addr.s_addr = inet_addr(IPADDR);
    if(INADDR_NONE == servaddr.sin_addr.s_addr) {
      close(sockfd);
      return 2;
    }

    if(connect(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) ==
       -1) {
      close(sockfd);
      printf("client error: connect: %s\n", strerror(errno));
      return 1;
    }

    /* no progress meter please */
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);

    /* send all data to this function  */
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);

    /* call this function to get a socket */
    curl_easy_setopt(curl, CURLOPT_OPENSOCKETFUNCTION, opensocket);
    curl_easy_setopt(curl, CURLOPT_OPENSOCKETDATA, &sockfd);

    /* call this function to close sockets */
    curl_easy_setopt(curl, CURLOPT_CLOSESOCKETFUNCTION, closecb);
    curl_easy_setopt(curl, CURLOPT_CLOSESOCKETDATA, &sockfd);

    /* call this function to set options for the socket */
    curl_easy_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_callback);

    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);

    res = curl_easy_perform(curl);

    curl_easy_cleanup(curl);

    close(sockfd);

    if(res) {
      printf("libcurl error: %d\n", res);
      return 4;
    }
  }

#ifdef _WIN32
  WSACleanup();
#endif
  return 0;
}
