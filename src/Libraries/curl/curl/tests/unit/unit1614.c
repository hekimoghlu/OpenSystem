/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
#include "curlcheck.h"

#include "noproxy.h"

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

struct check {
  const char *a;
  const char *n;
  unsigned int bits;
  bool match;
};

struct noproxy {
  const char *a;
  const char *n;
  bool match;
  bool space; /* space separated */
};

UNITTEST_START
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_PROXY)
{
  int i;
  int err = 0;
  struct check list4[]= {
    { "192.160.0.1", "192.160.0.1", 33, FALSE},
    { "192.160.0.1", "192.160.0.1", 32, TRUE},
    { "192.160.0.1", "192.160.0.1", 0, TRUE},
    { "192.160.0.1", "192.160.0.1", 24, TRUE},
    { "192.160.0.1", "192.160.0.1", 26, TRUE},
    { "192.160.0.1", "192.160.0.1", 20, TRUE},
    { "192.160.0.1", "192.160.0.1", 18, TRUE},
    { "192.160.0.1", "192.160.0.1", 12, TRUE},
    { "192.160.0.1", "192.160.0.1", 8, TRUE},
    { "192.160.0.1", "10.0.0.1", 8, FALSE},
    { "192.160.0.1", "10.0.0.1", 32, FALSE},
    { "192.160.0.1", "10.0.0.1", 0, FALSE},
    { NULL, NULL, 0, FALSE} /* end marker */
  };
  struct check list6[]= {
    { "::1", "::1", 0, TRUE},
    { "::1", "::1", 128, TRUE},
    { "::1", "0:0::1", 128, TRUE},
    { "::1", "0:0::1", 129, FALSE},
    { "fe80::ab47:4396:55c9:8474", "fe80::ab47:4396:55c9:8474", 64, TRUE},
    { NULL, NULL, 0, FALSE} /* end marker */
  };
  struct noproxy list[]= {
    { "www.example.com", "localhost .example.com .example.de", TRUE, TRUE},
    { "www.example.com", "localhost,.example.com,.example.de", TRUE, FALSE},
    { "www.example.com.", "localhost,.example.com,.example.de", TRUE, FALSE},
    { "example.com", "localhost,.example.com,.example.de", TRUE, FALSE},
    { "example.com.", "localhost,.example.com,.example.de", TRUE, FALSE},
    { "www.example.com", "localhost,.example.com.,.example.de", TRUE, FALSE},
    { "www.example.com", "localhost,www.example.com.,.example.de",
      TRUE, FALSE},
    { "example.com", "localhost,example.com,.example.de", TRUE, FALSE},
    { "example.com.", "localhost,example.com,.example.de", TRUE, FALSE},
    { "nexample.com", "localhost,example.com,.example.de", FALSE, FALSE},
    { "www.example.com", "localhost,example.com,.example.de", TRUE, FALSE},
    { "127.0.0.1", "127.0.0.1,localhost", TRUE, FALSE},
    { "127.0.0.1", "127.0.0.1,localhost,", TRUE, FALSE},
    { "127.0.0.1", "127.0.0.1/8,localhost,", TRUE, FALSE},
    { "127.0.0.1", "127.0.0.1/28,localhost,", TRUE, FALSE},
    { "127.0.0.1", "127.0.0.1/31,localhost,", TRUE, FALSE},
    { "127.0.0.1", "localhost,127.0.0.1", TRUE, FALSE},
    { "127.0.0.1", "localhost,127.0.0.1.127.0.0.1.127.0.0.1.127.0.0.1."
      "127.0.0.1.127.0.0.1.127.0.0.1.127.0.0.1.127.0.0.1.127.0.0.1.127."
      "0.0.1.127.0.0.1.127.0.0." /* 128 bytes "address" */, FALSE, FALSE},
    { "127.0.0.1", "localhost,127.0.0.1.127.0.0.1.127.0.0.1.127.0.0.1."
      "127.0.0.1.127.0.0.1.127.0.0.1.127.0.0.1.127.0.0.1.127.0.0.1.127."
      "0.0.1.127.0.0.1.127.0.0" /* 127 bytes "address" */, FALSE, FALSE},
    { "localhost", "localhost,127.0.0.1", TRUE, FALSE},
    { "localhost", "127.0.0.1,localhost", TRUE, FALSE},
    { "foobar", "barfoo", FALSE, FALSE},
    { "foobar", "foobar", TRUE, FALSE},
    { "192.168.0.1", "foobar", FALSE, FALSE},
    { "192.168.0.1", "192.168.0.0/16", TRUE, FALSE},
    { "192.168.0.1", "192.168.0.0/24", TRUE, FALSE},
    { "192.168.0.1", "192.168.0.0/32", FALSE, FALSE},
    { "192.168.0.1", "192.168.0.0", FALSE, FALSE},
    { "192.168.1.1", "192.168.0.0/24", FALSE, FALSE},
    { "192.168.1.1", "foo, bar, 192.168.0.0/24", FALSE, FALSE},
    { "192.168.1.1", "foo, bar, 192.168.0.0/16", TRUE, FALSE},
    { "[::1]", "foo, bar, 192.168.0.0/16", FALSE, FALSE},
    { "[::1]", "foo, bar, ::1/64", TRUE, FALSE},
    { "bar", "foo, bar, ::1/64", TRUE, FALSE},
    { "BAr", "foo, bar, ::1/64", TRUE, FALSE},
    { "BAr", "foo,,,,,              bar, ::1/64", TRUE, FALSE},
    { "www.example.com", "foo, .example.com", TRUE, FALSE},
    { "www.example.com", "www2.example.com, .example.net", FALSE, FALSE},
    { "example.com", ".example.com, .example.net", TRUE, FALSE},
    { "nonexample.com", ".example.com, .example.net", FALSE, FALSE},
    { NULL, NULL, FALSE, FALSE}
  };
  for(i = 0; list4[i].a; i++) {
    bool match = Curl_cidr4_match(list4[i].a, list4[i].n, list4[i].bits);
    if(match != list4[i].match) {
      fprintf(stderr, "%s in %s/%u should %smatch\n",
              list4[i].a, list4[i].n, list4[i].bits,
              list4[i].match ? "": "not ");
      err++;
    }
  }
  for(i = 0; list6[i].a; i++) {
    bool match = Curl_cidr6_match(list6[i].a, list6[i].n, list6[i].bits);
    if(match != list6[i].match) {
      fprintf(stderr, "%s in %s/%u should %smatch\n",
              list6[i].a, list6[i].n, list6[i].bits,
              list6[i].match ? "": "not ");
      err++;
    }
  }
  for(i = 0; list[i].a; i++) {
    bool spacesep = FALSE;
    bool match = Curl_check_noproxy(list[i].a, list[i].n, &spacesep);
    if(match != list[i].match) {
      fprintf(stderr, "%s in %s should %smatch\n",
              list[i].a, list[i].n,
              list[i].match ? "": "not ");
      err++;
    }
    if(spacesep != list[i].space) {
      fprintf(stderr, "%s is claimed to be %sspace separated\n",
              list[i].n, list[i].space?"":"NOT ");
      err++;
    }
  }
  fail_if(err, "errors");
}
#endif
UNITTEST_STOP
