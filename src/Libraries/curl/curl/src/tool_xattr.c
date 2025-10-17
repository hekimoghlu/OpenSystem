/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#include "tool_setup.h"
#include "tool_xattr.h"

#include "memdebug.h" /* keep this as LAST include */

#ifdef USE_XATTR

/* mapping table of curl metadata to extended attribute names */
static const struct xattr_mapping {
  const char *attr; /* name of the xattr */
  CURLINFO info;
} mappings[] = {
  /* mappings proposed by
   * https://freedesktop.org/wiki/CommonExtendedAttributes/
   */
  { "user.xdg.referrer.url", CURLINFO_REFERER },
  { "user.mime_type",        CURLINFO_CONTENT_TYPE },
  { NULL,                    CURLINFO_NONE } /* last element, abort here */
};

/* returns a new URL that needs to be freed */
/* @unittest: 1621 */
#ifdef UNITTESTS
char *stripcredentials(const char *url);
#else
static
#endif
char *stripcredentials(const char *url)
{
  CURLU *u;
  CURLUcode uc;
  char *nurl;
  u = curl_url();
  if(u) {
    uc = curl_url_set(u, CURLUPART_URL, url, 0);
    if(uc)
      goto error;

    uc = curl_url_set(u, CURLUPART_USER, NULL, 0);
    if(uc)
      goto error;

    uc = curl_url_set(u, CURLUPART_PASSWORD, NULL, 0);
    if(uc)
      goto error;

    uc = curl_url_get(u, CURLUPART_URL, &nurl, 0);
    if(uc)
      goto error;

    curl_url_cleanup(u);

    return nurl;
  }
error:
  curl_url_cleanup(u);
  return NULL;
}

static int xattr(int fd,
                 const char *attr, /* name of the xattr */
                 const char *value)
{
  int err = 0;
  if(value) {
#ifdef DEBUGBUILD
    (void)fd;
    if(getenv("CURL_FAKE_XATTR")) {
      printf("%s => %s\n", attr, value);
    }
    return 0;
#else
#ifdef HAVE_FSETXATTR_6
    err = fsetxattr(fd, attr, value, strlen(value), 0, 0);
#elif defined(HAVE_FSETXATTR_5)
    err = fsetxattr(fd, attr, value, strlen(value), 0);
#elif defined(__FreeBSD_version) || defined(__MidnightBSD_version)
    {
      ssize_t rc = extattr_set_fd(fd, EXTATTR_NAMESPACE_USER,
                                  attr, value, strlen(value));
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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
 */err = (rc < 0 ? -1 : 0);
    }
#endif
#endif
  }
  return err;
}
/* store metadata from the curl request alongside the downloaded
 * file using extended attributes
 */
int fwrite_xattr(CURL *curl, const char *url, int fd)
{
  int i = 0;
  int err = 0;

  /* loop through all xattr-curlinfo pairs and abort on a set error */
  while(!err && mappings[i].attr) {
    char *value = NULL;
    CURLcode result = curl_easy_getinfo(curl, mappings[i].info, &value);
    if(!result && value)
      err = xattr(fd, mappings[i].attr, value);
    i++;
  }
  if(!err) {
    char *nurl = stripcredentials(url);
    if(!nurl)
      return 1;
    err = xattr(fd, "user.xdg.origin.url", nurl);
    curl_free(nurl);
  }
  return err;
}
#endif

