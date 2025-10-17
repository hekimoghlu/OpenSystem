/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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

#ifndef CURLINC_WEBSOCKETS_H
#define CURLINC_WEBSOCKETS_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/

#ifdef  __cplusplus
extern "C" {
#endif

struct curl_ws_frame {
  int age;              /* zero */
  int flags;            /* See the CURLWS_* defines */
  curl_off_t offset;    /* the offset of this data into the frame */
  curl_off_t bytesleft; /* number of pending bytes left of the payload */
  size_t len;           /* size of the current data chunk */
};

/* flag bits */
#define CURLWS_TEXT       (1<<0)
#define CURLWS_BINARY     (1<<1)
#define CURLWS_CONT       (1<<2)
#define CURLWS_CLOSE      (1<<3)
#define CURLWS_PING       (1<<4)
#define CURLWS_OFFSET     (1<<5)

/*
 * NAME curl_ws_recv()
 *
 * DESCRIPTION
 *
 * Receives data from the websocket connection. Use after successful
 * curl_easy_perform() with CURLOPT_CONNECT_ONLY option.
 */
CURL_EXTERN CURLcode curl_ws_recv(CURL *curl, void *buffer, size_t buflen,
                                  size_t *recv,
                                  const struct curl_ws_frame **metap);

/* flags for curl_ws_send() */
#define CURLWS_PONG       (1<<6)

/*
 * NAME curl_ws_send()
 *
 * DESCRIPTION
 *
 * Sends data over the websocket connection. Use after successful
 * curl_easy_perform() with CURLOPT_CONNECT_ONLY option.
 */
CURL_EXTERN CURLcode curl_ws_send(CURL *curl, const void *buffer,
                                  size_t buflen, size_t *sent,
                                  curl_off_t fragsize,
                                  unsigned int flags);

/* bits for the CURLOPT_WS_OPTIONS bitmask: */
#define CURLWS_RAW_MODE (1<<0)

CURL_EXTERN const struct curl_ws_frame *curl_ws_meta(CURL *curl);

#ifdef  __cplusplus
}
#endif

#endif /* CURLINC_WEBSOCKETS_H */
