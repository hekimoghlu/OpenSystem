/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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
 * Verify an SMTP email address
 * </DESC>
 */

#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

/* This is a simple example showing how to verify an email address from an
 * SMTP server.
 *
 * Notes:
 *
 * 1) This example requires libcurl 7.34.0 or above.
 * 2) Not all email servers support this command and even if your email server
 *    does support it, it may respond with a 252 response code even though the
 *    address does not exist.
 */

int main(void)
{
  CURL *curl;
  CURLcode res;
  struct curl_slist *recipients = NULL;

  curl = curl_easy_init();
  if(curl) {
    /* This is the URL for your mailserver */
    curl_easy_setopt(curl, CURLOPT_URL, "smtp://mail.example.com");

    /* Note that the CURLOPT_MAIL_RCPT takes a list, not a char array  */
    recipients = curl_slist_append(recipients, "<recipient@example.com>");
    curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);

    /* Perform the VRFY */
    res = curl_easy_perform(curl);

    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    /* Free the list of recipients */
    curl_slist_free_all(recipients);

    /* curl does not send the QUIT command until you call cleanup, so you
     * should be able to reuse this connection for additional requests. It may
     * not be a good idea to keep the connection open for a long time though
     * (more than a few minutes may result in the server timing out the
     * connection) and you do want to clean up in the end.
     */
    curl_easy_cleanup(curl);
  }

  return 0;
}
