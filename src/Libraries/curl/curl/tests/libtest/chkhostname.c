/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#include "curl_setup.h"

#include "curl_gethostname.h"

#define HOSTNAME_MAX 1024

int main(int argc, char *argv[])
{
  char buff[HOSTNAME_MAX];
  if(argc != 2) {
    printf("Usage: %s EXPECTED_HOSTNAME\n", argv[0]);
    return 1;
  }

  if(Curl_gethostname(buff, HOSTNAME_MAX)) {
    printf("Curl_gethostname() failed\n");
    return 1;
  }

  /* compare the name returned by Curl_gethostname() with the expected one */
  if(strncmp(buff, argv[1], HOSTNAME_MAX)) {
    printf("got unexpected host name back, LD_PRELOAD failed\n");
    return 1;
  }
  return 0;
}
