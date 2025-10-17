/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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
 * Retrieve emails from a shared POP3 mailbox
 * </DESC>
 */

#include <stdio.h>
#include <curl/curl.h>

/* This is a simple example showing how to retrieve mail using libcurl's POP3
 * capabilities.
 *
 * Note that this example requires libcurl 7.66.0 or above.
 */

int main(void)
{
  CURL *curl;
  CURLcode res = CURLE_OK;

  curl = curl_easy_init();
  if(curl) {
    /* Set the username and password */
    curl_easy_setopt(curl, CURLOPT_USERNAME, "user");
    curl_easy_setopt(curl, CURLOPT_PASSWORD, "secret");

    /* Set the authorization identity (identity to act as) */
    curl_easy_setopt(curl, CURLOPT_SASL_AUTHZID, "shared-mailbox");

    /* Force PLAIN authentication */
    curl_easy_setopt(curl, CURLOPT_LOGIN_OPTIONS, "AUTH=PLAIN");

    /* This retrieves message 1 from the user's mailbox */
    curl_easy_setopt(curl, CURLOPT_URL, "pop3://pop.example.com/1");

    /* Perform the retr */
    res = curl_easy_perform(curl);

    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    /* Always cleanup */
    curl_easy_cleanup(curl);
  }

  return (int)res;
}
