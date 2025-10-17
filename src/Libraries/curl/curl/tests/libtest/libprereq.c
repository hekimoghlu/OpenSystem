/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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

typedef struct prcs {
  int prereq_retcode;
  int ipv6;
} PRCS;

static int prereq_callback(void *clientp,
                           char *conn_primary_ip,
                           char *conn_local_ip,
                           int conn_primary_port,
                           int conn_local_port)
{
  PRCS *prereq_cb = (PRCS *)clientp;

  if(prereq_cb->ipv6) {
    printf("Connected to [%s]\n", conn_primary_ip);
    printf("Connected from [%s]\n", conn_local_ip);
  }
  else {
    printf("Connected to %s\n", conn_primary_ip);
    printf("Connected from %s\n", conn_local_ip);
  }

  printf("Remote port = %d\n", conn_primary_port);
  printf("Local port = %d\n", conn_local_port);
  printf("Returning = %d\n", prereq_cb->prereq_retcode);
  return prereq_cb->prereq_retcode;
}

int test(char *URL)
{
  PRCS prereq_cb;
  CURLcode ret = CURLE_OK;
  CURL *curl = NULL;

  prereq_cb.prereq_retcode = CURL_PREREQFUNC_OK;
  prereq_cb.ipv6 = 0;

  curl_global_init(CURL_GLOBAL_ALL);
  curl = curl_easy_init();

  if(curl) {
    if(strstr(URL, "#ipv6")) {
      /* The IP addresses should be surrounded by brackets! */
      prereq_cb.ipv6 = 1;
    }
    if(strstr(URL, "#err")) {
      /* Set the callback to exit with failure */
      prereq_cb.prereq_retcode = CURL_PREREQFUNC_ABORT;
    }

    curl_easy_setopt(curl, CURLOPT_URL, URL);
    curl_easy_setopt(curl, CURLOPT_PREREQFUNCTION, prereq_callback);
    curl_easy_setopt(curl, CURLOPT_PREREQDATA, &prereq_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, stderr);

    if(strstr(URL, "#redir")) {
      /* Enable follow-location */
      curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
    }

    ret = curl_easy_perform(curl);
    if(ret) {
      fprintf(stderr, "%s:%d curl_easy_perform() failed with code %d (%s)\n",
          __FILE__, __LINE__, ret, curl_easy_strerror(ret));
      goto test_cleanup;
    }
  }

test_cleanup:
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return ret;
}
