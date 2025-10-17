/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
 * FTP upload a file from memory
 * </DESC>
 */
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>

static const char data[]=
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
  "Nam rhoncus odio id venenatis volutpat. Vestibulum dapibus "
  "bibendum ullamcorper. Maecenas finibus elit augue, vel "
  "condimentum odio maximus nec. In hac habitasse platea dictumst. "
  "Vestibulum vel dolor et turpis rutrum finibus ac at nulla. "
  "Vivamus nec neque ac elit blandit pretium vitae maximus ipsum. "
  "Quisque sodales magna vel erat auctor, sed pellentesque nisi "
  "rhoncus. Donec vehicula maximus pretium. Aliquam eu tincidunt "
  "lorem.";

struct WriteThis {
  const char *readptr;
  size_t sizeleft;
};

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *userp)
{
  struct WriteThis *upload = (struct WriteThis *)userp;
  size_t max = size*nmemb;

  if(max < 1)
    return 0;

  if(upload->sizeleft) {
    size_t copylen = max;
    if(copylen > upload->sizeleft)
      copylen = upload->sizeleft;
    memcpy(ptr, upload->readptr, copylen);
    upload->readptr += copylen;
    upload->sizeleft -= copylen;
    return copylen;
  }

  return 0;                          /* no more data left to deliver */
}

int main(void)
{
  CURL *curl;
  CURLcode res;

  struct WriteThis upload;

  upload.readptr = data;
  upload.sizeleft = strlen(data);

  /* In windows, this inits the winsock stuff */
  res = curl_global_init(CURL_GLOBAL_DEFAULT);
  /* Check for errors */
  if(res != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed: %s\n",
            curl_easy_strerror(res));
    return 1;
  }

  /* get a curl handle */
  curl = curl_easy_init();
  if(curl) {
    /* First set the URL, the target file */
    curl_easy_setopt(curl, CURLOPT_URL,
                     "ftp://example.com/path/to/upload/file");

    /* User and password for the FTP login */
    curl_easy_setopt(curl, CURLOPT_USERPWD, "login:secret");

    /* Now specify we want to UPLOAD data */
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);

    /* we want to use our own read function */
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_callback);

    /* pointer to pass to our read function */
    curl_easy_setopt(curl, CURLOPT_READDATA, &upload);

    /* get verbose debug output please */
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    /* Set the expected upload size. */
    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE,
                     (curl_off_t)upload.sizeleft);

    /* Perform the request, res gets the return code */
    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    /* always cleanup */
    curl_easy_cleanup(curl);
  }
  curl_global_cleanup();
  return 0;
}
