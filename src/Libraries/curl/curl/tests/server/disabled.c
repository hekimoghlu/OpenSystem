/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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
/*
 * The purpose of this tool is to figure out which, if any, features that are
 * disabled which should otherwise exist and work. These aren't visible in
 * regular curl -V output.
 *
 * Disabled protocols are visible in curl_version_info() and are not included
 * in this table.
 */

#include "curl_setup.h"
#include "multihandle.h" /* for ENABLE_WAKEUP */
#include "tool_xattr.h" /* for USE_XATTR */
#include "curl_sha512_256.h" /* for CURL_HAVE_SHA512_256 */
#include <stdio.h>

static const char *disabled[]={
#ifdef CURL_DISABLE_BINDLOCAL
  "bindlocal",
#endif
#ifdef CURL_DISABLE_COOKIES
  "cookies",
#endif
#ifdef CURL_DISABLE_BASIC_AUTH
  "basic-auth",
#endif
#ifdef CURL_DISABLE_BEARER_AUTH
  "bearer-auth",
#endif
#ifdef CURL_DISABLE_DIGEST_AUTH
  "digest-auth",
#endif
#ifdef CURL_DISABLE_NEGOTIATE_AUTH
  "negotiate-auth",
#endif
#ifdef CURL_DISABLE_AWS
  "aws",
#endif
#ifdef CURL_DISABLE_DOH
  "DoH",
#endif
#ifdef CURL_DISABLE_HTTP_AUTH
  "HTTP-auth",
#endif
#ifdef CURL_DISABLE_MIME
  "Mime",
#endif
#ifdef CURL_DISABLE_NETRC
  "netrc",
#endif
#ifdef CURL_DISABLE_PARSEDATE
  "parsedate",
#endif
#ifdef CURL_DISABLE_PROXY
  "proxy",
#endif
#ifdef CURL_DISABLE_SHUFFLE_DNS
  "shuffle-dns",
#endif
#ifdef CURL_DISABLE_TYPECHECK
  "typecheck",
#endif
#ifdef CURL_DISABLE_VERBOSE_STRINGS
  "verbose-strings",
#endif
#ifndef ENABLE_WAKEUP
  "wakeup",
#endif
#ifdef CURL_DISABLE_HEADERS_API
  "headers-api",
#endif
#ifndef USE_XATTR
  "xattr",
#endif
#ifdef CURL_DISABLE_FORM_API
  "form-api",
#endif
#if (SIZEOF_TIME_T < 5)
  "large-time",
#endif
#ifndef CURL_HAVE_SHA512_256
  "sha512-256",
#endif
  NULL
};

int main(int argc, char **argv)
{
  int i;

  (void) argc;
  (void) argv;

  for(i = 0; disabled[i]; i++)
    printf("%s\n", disabled[i]);

  return 0;
}
