/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
 * Send email with IMAP
 * </DESC>
 */

#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

/* This is a simple example showing how to send mail using libcurl's IMAP
 * capabilities.
 *
 * Note that this example requires libcurl 7.30.0 or above.
 */

#define FROM    "<sender@example.org>"
#define TO      "<addressee@example.net>"
#define CC      "<info@example.org>"

static const char *payload_text =
  "Date: Mon, 29 Nov 2010 21:54:29 +1100\r\n"
  "To: " TO "\r\n"
  "From: " FROM "(Example User)\r\n"
  "Cc: " CC "(Another example User)\r\n"
  "Message-ID: "
  "<dcd7cb36-11db-487a-9f3a-e652a9458efd@rfcpedant.example.org>\r\n"
  "Subject: IMAP example message\r\n"
  "\r\n" /* empty line to divide headers from body, see RFC 5322 */
  "The body of the message starts here.\r\n"
  "\r\n"
  "It could be a lot of lines, could be MIME encoded, whatever.\r\n"
  "Check RFC 5322.\r\n";

struct upload_status {
  size_t bytes_read;
};

static size_t payload_source(char *ptr, size_t size, size_t nmemb, void *userp)
{
  struct upload_status *upload_ctx = (struct upload_status *)userp;
  const char *data;
  size_t room = size * nmemb;

  if((size == 0) || (nmemb == 0) || ((size*nmemb) < 1)) {
    return 0;
  }

  data = &payload_text[upload_ctx->bytes_read];

  if(*data) {
    size_t len = strlen(data);
    if(room < len)
      len = room;
    memcpy(ptr, data, len);
    upload_ctx->bytes_read += len;

    return len;
  }

  return 0;
}

int main(void)
{
  CURL *curl;
  CURLcode res = CURLE_OK;

  curl = curl_easy_init();
  if(curl) {
    size_t filesize;
    long infilesize = LONG_MAX;
    struct upload_status upload_ctx = { 0 };

    /* Set username and password */
    curl_easy_setopt(curl, CURLOPT_USERNAME, "user");
    curl_easy_setopt(curl, CURLOPT_PASSWORD, "secret");

    /* This creates a new message in folder "Sent". */
    curl_easy_setopt(curl, CURLOPT_URL, "imap://imap.example.com/Sent");

    /* In this case, we are using a callback function to specify the data. You
     * could just use the CURLOPT_READDATA option to specify a FILE pointer to
     * read from. */
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, payload_source);
    curl_easy_setopt(curl, CURLOPT_READDATA, &upload_ctx);
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);

    filesize = strlen(payload_text);
    if(filesize <= LONG_MAX)
      infilesize = (long)filesize;
    curl_easy_setopt(curl, CURLOPT_INFILESIZE, infilesize);

    /* Perform the append */
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
