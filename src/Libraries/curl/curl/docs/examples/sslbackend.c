/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
 * Shows HTTPS usage with client certs and optional ssl engine use.
 * </DESC>
 */
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <curl/curl.h>

/*
 * An SSL-enabled libcurl is required for this sample to work (at least one
 * SSL backend has to be configured).
 *
 *  **** This example only works with libcurl 7.56.0 and later! ****
*/

int main(int argc, char **argv)
{
  const char *name = argc > 1 ? argv[1] : "openssl";
  CURLsslset result;

  if(!strcmp("list", name)) {
    const curl_ssl_backend **list;
    int i;

    result = curl_global_sslset(CURLSSLBACKEND_NONE, NULL, &list);
    assert(result == CURLSSLSET_UNKNOWN_BACKEND);

    for(i = 0; list[i]; i++)
      printf("SSL backend #%d: '%s' (ID: %d)\n",
             i, list[i]->name, list[i]->id);

    return 0;
  }
  else if(isdigit((int)(unsigned char)*name)) {
    int id = atoi(name);

    result = curl_global_sslset((curl_sslbackend)id, NULL, NULL);
  }
  else
    result = curl_global_sslset(CURLSSLBACKEND_NONE, name, NULL);

  if(result == CURLSSLSET_UNKNOWN_BACKEND) {
    fprintf(stderr, "Unknown SSL backend id: %s\n", name);
    return 1;
  }

  assert(result == CURLSSLSET_OK);

  printf("Version with SSL backend '%s':\n\n\t%s\n", name, curl_version());

  return 0;
}
