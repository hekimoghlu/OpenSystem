/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
 * Gets a file using an SFTP URL.
 * </DESC>
 */

#include <stdio.h>

#include <curl/curl.h>

/* define this to switch off the use of ssh-agent in this program */
#undef DISABLE_SSH_AGENT

/*
 * This is an example showing how to get a single file from an SFTP server. It
 * delays the actual destination file creation until the first write callback
 * so that it does not create an empty file in case the remote file does not
 * exist or something else fails.
 */

struct FtpFile {
  const char *filename;
  FILE *stream;
};

static size_t my_fwrite(void *buffer, size_t size, size_t nmemb,
                        void *stream)
{
  struct FtpFile *out = (struct FtpFile *)stream;
  if(!out->stream) {
    /* open file for writing */
    out->stream = fopen(out->filename, "wb");
    if(!out->stream)
      return 0; /* failure, cannot open file to write */
  }
  return fwrite(buffer, size, nmemb, out->stream);
}


int main(void)
{
  CURL *curl;
  CURLcode res;
  struct FtpFile ftpfile = {
    "yourfile.bin", /* name to store the file as if successful */
    NULL
  };

  curl_global_init(CURL_GLOBAL_DEFAULT);

  curl = curl_easy_init();
  if(curl) {
    /*
     * You better replace the URL with one that works!
     */
    curl_easy_setopt(curl, CURLOPT_URL,
                     "sftp://user@server/home/user/file.txt");
    /* Define our callback to get called when there is data to be written */
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
    /* Set a pointer to our struct to pass to the callback */
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ftpfile);

#ifndef DISABLE_SSH_AGENT
    /* We activate ssh agent. For this to work you need
       to have ssh-agent running (type set | grep SSH_AGENT to check) or
       pageant on Windows (there is an icon in systray if so) */
    curl_easy_setopt(curl, CURLOPT_SSH_AUTH_TYPES, CURLSSH_AUTH_AGENT);
#endif

    /* Switch on full protocol/debug output */
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    res = curl_easy_perform(curl);

    /* always cleanup */
    curl_easy_cleanup(curl);

    if(CURLE_OK != res) {
      /* we failed */
      fprintf(stderr, "curl told us %d\n", res);
    }
  }

  if(ftpfile.stream)
    fclose(ftpfile.stream); /* close the local file */

  curl_global_cleanup();

  return 0;
}
