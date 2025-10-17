/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
 * WebSocket using CONNECT_ONLY
 * </DESC>
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <curl/curl.h>

static int ping(CURL *curl, const char *send_payload)
{
  size_t sent;
  CURLcode result =
    curl_ws_send(curl, send_payload, strlen(send_payload), &sent, 0,
                 CURLWS_PING);
  return (int)result;
}

static int recv_pong(CURL *curl, const char *expected_payload)
{
  size_t rlen;
  const struct curl_ws_frame *meta;
  char buffer[256];
  CURLcode result = curl_ws_recv(curl, buffer, sizeof(buffer), &rlen, &meta);
  if(!result) {
    if(meta->flags & CURLWS_PONG) {
      int same = 0;
      fprintf(stderr, "ws: got PONG back\n");
      if(rlen == strlen(expected_payload)) {
        if(!memcmp(expected_payload, buffer, rlen)) {
          fprintf(stderr, "ws: got the same payload back\n");
          same = 1;
        }
      }
      if(!same)
        fprintf(stderr, "ws: did NOT get the same payload back\n");
    }
    else {
      fprintf(stderr, "recv_pong: got %u bytes rflags %x\n", (int)rlen,
              meta->flags);
    }
  }
  fprintf(stderr, "ws: curl_ws_recv returned %u, received %u\n",
          (unsigned int)result, (unsigned int)rlen);
  return (int)result;
}

static int recv_any(CURL *curl)
{
  size_t rlen;
  const struct curl_ws_frame *meta;
  char buffer[256];
  CURLcode result = curl_ws_recv(curl, buffer, sizeof(buffer), &rlen, &meta);
  if(result)
    return result;

  return 0;
}

/* close the connection */
static void websocket_close(CURL *curl)
{
  size_t sent;
  (void)curl_ws_send(curl, "", 0, &sent, 0, CURLWS_CLOSE);
}

static void websocket(CURL *curl)
{
  int i = 0;
  do {
    recv_any(curl);
    if(ping(curl, "foobar"))
      return;
    if(recv_pong(curl, "foobar")) {
      return;
    }
    sleep(2);
  } while(i++ < 10);
  websocket_close(curl);
}

int main(void)
{
  CURL *curl;
  CURLcode res;

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, "wss://example.com");

    curl_easy_setopt(curl, CURLOPT_CONNECT_ONLY, 2L); /* websocket style */

    /* Perform the request, res gets the return code */
    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));
    else {
      /* connected and ready */
      websocket(curl);
    }

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  return 0;
}
