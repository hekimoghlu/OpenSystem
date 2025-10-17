/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
 * Include necessary headers...
 */

#include "cups-private.h"
#include "debug-internal.h"
#include <fcntl.h>
#include <math.h>
#ifdef _WIN32
#  include <tchar.h>
#else
#  include <signal.h>
#  include <sys/time.h>
#  include <sys/resource.h>
#endif /* _WIN32 */
#ifdef HAVE_POLL
#  include <poll.h>
#endif /* HAVE_POLL */


/*
 * Include platform-specific TLS code...
 */

#ifdef HAVE_SSL
#  ifdef HAVE_GNUTLS
#    include "tls-gnutls.c"
#  elif defined(HAVE_CDSASSL)
#    include "tls-darwin.c"
#  elif defined(HAVE_SSPISSL)
#    include "tls-sspi.c"
#  endif /* HAVE_GNUTLS */
#else
/* Stubs for when TLS is not supported/available */
int
httpCopyCredentials(http_t *http, cups_array_t **credentials)
{
  (void)http;
  if (credentials)
    *credentials = NULL;
  return (-1);
}
int
httpCredentialsAreValidForName(cups_array_t *credentials, const char *common_name)
{
  (void)credentials;
  (void)common_name;
  return (1);
}
time_t
httpCredentialsGetExpiration(cups_array_t *credentials)
{
  (void)credentials;
  return (INT_MAX);
}
http_trust_t
httpCredentialsGetTrust(cups_array_t *credentials, const char *common_name)
{
  (void)credentials;
  (void)common_name;
  return (HTTP_TRUST_OK);
}
size_t
httpCredentialsString(cups_array_t *credentials, char *buffer, size_t bufsize)
{
  (void)credentials;
  (void)bufsize;
  if (buffer)
    *buffer = '\0';
  return (0);
}
int
httpLoadCredentials(const char *path, cups_array_t **credentials, const char *common_name)
{
  (void)path;
  (void)credentials;
  (void)common_name;
  return (-1);
}
int
httpSaveCredentials(const char *path, cups_array_t *credentials, const char *common_name)
{
  (void)path;
  (void)credentials;
  (void)common_name;
  return (-1);
}
#endif /* HAVE_SSL */
