/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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

#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

static int loadfile(const char *filename, void **filedata, size_t *filesize)
{
  size_t datasize = 0;
  void *data = NULL;
  if(filename) {
    FILE *fInCert = fopen(filename, "rb");

    if(fInCert) {
      long cert_tell = 0;
      bool continue_reading = fseek(fInCert, 0, SEEK_END) == 0;
      if(continue_reading)
        cert_tell = ftell(fInCert);
      if(cert_tell < 0)
        continue_reading = FALSE;
      else
        datasize = (size_t)cert_tell;
      if(continue_reading)
        continue_reading = fseek(fInCert, 0, SEEK_SET) == 0;
      if(continue_reading)
        data = malloc(datasize + 1);
      if((!data) ||
         ((int)fread(data, datasize, 1, fInCert) != 1))
        continue_reading = FALSE;
      fclose(fInCert);
      if(!continue_reading) {
        free(data);
        datasize = 0;
        data = NULL;
      }
   }
  }
  *filesize = datasize;
  *filedata = data;
  return data ? 1 : 0;
}

static int test_cert_blob(const char *url, const char *cafile)
{
  CURLcode code = CURLE_OUT_OF_MEMORY;
  CURL *curl;
  struct curl_blob blob;
  size_t certsize;
  void *certdata;

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    return CURLE_FAILED_INIT;
  }

  if(loadfile(cafile, &certdata, &certsize)) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE,     1L);
    curl_easy_setopt(curl, CURLOPT_HEADER,      1L);
    curl_easy_setopt(curl, CURLOPT_URL,         url);
    curl_easy_setopt(curl, CURLOPT_USERAGENT,   "CURLOPT_CAINFO_BLOB");
    curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS,
                     CURLSSLOPT_REVOKE_BEST_EFFORT);

    blob.data = certdata;
    blob.len = certsize;
    blob.flags = CURL_BLOB_COPY;
    curl_easy_setopt(curl, CURLOPT_CAINFO_BLOB, &blob);
    free(certdata);
    code = curl_easy_perform(curl);
  }
  curl_easy_cleanup(curl);

  return (int)code;
}

int test(char *URL)
{
  int res = 0;
  curl_global_init(CURL_GLOBAL_DEFAULT);
  if(!strcmp("check", URL)) {
    CURL *e;
    CURLcode w = CURLE_OK;
    struct curl_blob blob = {0};
    e = curl_easy_init();
    if(e) {
      w = curl_easy_setopt(e, CURLOPT_CAINFO_BLOB, &blob);
      if(w)
        printf("CURLOPT_CAINFO_BLOB is not supported\n");
      curl_easy_cleanup(e);
    }
    res = (int)w;
  }
  else
    res = test_cert_blob(URL, libtest_arg2);

  curl_global_cleanup();
  return res;
}
