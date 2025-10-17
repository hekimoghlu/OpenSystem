/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

#if defined(USE_OPENSSL)                        \
  || defined(USE_SCHANNEL)
/* these backends use functions from this file */

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_IN6_H
#include <netinet/in6.h>
#endif
#include "curl_memrchr.h"

#include "hostcheck.h"
#include "strcase.h"
#include "hostip.h"

#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

/* check the two input strings with given length, but do not
   assume they end in nul-bytes */
static bool pmatch(const char *hostname, size_t hostlen,
                   const char *pattern, size_t patternlen)
{
  if(hostlen != patternlen)
    return FALSE;
  return strncasecompare(hostname, pattern, hostlen);
}

/*
 * Match a hostname against a wildcard pattern.
 * E.g.
 *  "foo.host.com" matches "*.host.com".
 *
 * We use the matching rule described in RFC6125, section 6.4.3.
 * https://datatracker.ietf.org/doc/html/rfc6125#section-6.4.3
 *
 * In addition: ignore trailing dots in the host names and wildcards, so that
 * the names are used normalized. This is what the browsers do.
 *
 * Do not allow wildcard matching on IP numbers. There are apparently
 * certificates being used with an IP address in the CN field, thus making no
 * apparent distinction between a name and an IP. We need to detect the use of
 * an IP address and not wildcard match on such names.
 *
 * Only match on "*" being used for the leftmost label, not "a*", "a*b" nor
 * "*b".
 *
 * Return TRUE on a match. FALSE if not.
 *
 * @unittest: 1397
 */

static bool hostmatch(const char *hostname,
                      size_t hostlen,
                      const char *pattern,
                      size_t patternlen)
{
  const char *pattern_label_end;

  DEBUGASSERT(pattern);
  DEBUGASSERT(patternlen);
  DEBUGASSERT(hostname);
  DEBUGASSERT(hostlen);

  /* normalize pattern and hostname by stripping off trailing dots */
  if(hostname[hostlen-1]=='.')
    hostlen--;
  if(pattern[patternlen-1]=='.')
    patternlen--;

  if(strncmp(pattern, "*.", 2))
    return pmatch(hostname, hostlen, pattern, patternlen);

  /* detect IP address as hostname and fail the match if so */
  else if(Curl_host_is_ipnum(hostname))
    return FALSE;

  /* We require at least 2 dots in the pattern to avoid too wide wildcard
     match. */
  pattern_label_end = memchr(pattern, '.', patternlen);
  if(!pattern_label_end ||
     (memrchr(pattern, '.', patternlen) == pattern_label_end))
    return pmatch(hostname, hostlen, pattern, patternlen);
  else {
    const char *hostname_label_end = memchr(hostname, '.', hostlen);
    if(hostname_label_end) {
      size_t skiphost = hostname_label_end - hostname;
      size_t skiplen = pattern_label_end - pattern;
      return pmatch(hostname_label_end, hostlen - skiphost,
                    pattern_label_end, patternlen - skiplen);
    }
  }
  return FALSE;
}

/*
 * Curl_cert_hostcheck() returns TRUE if a match and FALSE if not.
 */
bool Curl_cert_hostcheck(const char *match, size_t matchlen,
                         const char *hostname, size_t hostlen)
{
  if(match && *match && hostname && *hostname)
    return hostmatch(hostname, hostlen, match, matchlen);
  return FALSE;
}

#endif /* OPENSSL or SCHANNEL */
