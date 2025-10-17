/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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


static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{
}

/* only these backends define the tested functions */
#if defined(USE_OPENSSL) || defined(USE_SCHANNEL)
#include "vtls/hostcheck.h"
struct testcase {
  const char *host;
  const char *pattern;
  bool match;
};

static struct testcase tests[] = {
  {"", "", FALSE},
  {"a", "", FALSE},
  {"", "b", FALSE},
  {"a", "b", FALSE},
  {"aa", "bb", FALSE},
  {"\xff", "\xff", TRUE},
  {"aa.aa.aa", "aa.aa.bb", FALSE},
  {"aa.aa.aa", "aa.aa.aa", TRUE},
  {"aa.aa.aa", "*.aa.bb", FALSE},
  {"aa.aa.aa", "*.aa.aa", TRUE},
  {"192.168.0.1", "192.168.0.1", TRUE},
  {"192.168.0.1", "*.168.0.1", FALSE},
  {"192.168.0.1", "*.0.1", FALSE},
  {"h.ello", "*.ello", FALSE},
  {"h.ello.", "*.ello", FALSE},
  {"h.ello", "*.ello.", FALSE},
  {"h.e.llo", "*.e.llo", TRUE},
  {"h.e.llo", " *.e.llo", FALSE},
  {" h.e.llo", "*.e.llo", TRUE},
  {"h.e.llo.", "*.e.llo", TRUE},
  {"*.e.llo.", "*.e.llo", TRUE},
  {"************.e.llo.", "*.e.llo", TRUE},
  {"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
   "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
   "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
   "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
   "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
   ".e.llo.", "*.e.llo", TRUE},
  {"\xfe\xfe.e.llo.", "*.e.llo", TRUE},
  {"h.e.llo.", "*.e.llo.", TRUE},
  {"h.e.llo", "*.e.llo.", TRUE},
  {".h.e.llo", "*.e.llo.", FALSE},
  {"h.e.llo", "*.*.llo.", FALSE},
  {"h.e.llo", "h.*.llo", FALSE},
  {"h.e.llo", "h.e.*", FALSE},
  {"hello", "*.ello", FALSE},
  {"hello", "**llo", FALSE},
  {"bar.foo.example.com", "*.example.com", FALSE},
  {"foo.example.com", "*.example.com", TRUE},
  {"baz.example.net", "b*z.example.net", FALSE},
  {"foobaz.example.net", "*baz.example.net", FALSE},
  {"xn--l8j.example.local", "x*.example.local", FALSE},
  {"xn--l8j.example.net", "*.example.net", TRUE},
  {"xn--l8j.example.net", "*j.example.net", FALSE},
  {"xn--l8j.example.net", "xn--l8j.example.net", TRUE},
  {"xn--l8j.example.net", "xn--l8j.*.net", FALSE},
  {"xl8j.example.net", "*.example.net", TRUE},
  {"fe80::3285:a9ff:fe46:b619", "*::3285:a9ff:fe46:b619", FALSE},
  {"fe80::3285:a9ff:fe46:b619", "fe80::3285:a9ff:fe46:b619", TRUE},
  {NULL, NULL, FALSE}
};

UNITTEST_START
{
  int i;
  for(i = 0; tests[i].host; i++) {
    if(tests[i].match != Curl_cert_hostcheck(tests[i].pattern,
                                             strlen(tests[i].pattern),
                                             tests[i].host,
                                             strlen(tests[i].host))) {
      fprintf(stderr,
              "HOST: %s\n"
              "PTRN: %s\n"
              "did %sMATCH\n",
              tests[i].host,
              tests[i].pattern,
              tests[i].match ? "NOT ": "");
      unitfail++;
    }
  }
}

UNITTEST_STOP
#else

UNITTEST_START

UNITTEST_STOP
#endif
