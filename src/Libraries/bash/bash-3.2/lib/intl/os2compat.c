/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
#define OS2_AWARE
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

/* A version of getenv() that works from DLLs */
extern unsigned long DosScanEnv (const unsigned char *pszName, unsigned char **ppszValue);

char *
_nl_getenv (const char *name)
{
  unsigned char *value;
  if (DosScanEnv (name, &value))
    return NULL;
  else
    return value;
}

/* A fixed size buffer.  */
char libintl_nl_default_dirname[MAXPATHLEN+1];

char *_nlos2_libdir = NULL;
char *_nlos2_localealiaspath = NULL;
char *_nlos2_localedir = NULL;

static __attribute__((constructor)) void
nlos2_initialize ()
{
  char *root = getenv ("UNIXROOT");
  char *gnulocaledir = getenv ("GNULOCALEDIR");

  _nlos2_libdir = gnulocaledir;
  if (!_nlos2_libdir)
    {
      if (root)
        {
          size_t sl = strlen (root);
          _nlos2_libdir = (char *) malloc (sl + strlen (LIBDIR) + 1);
          memcpy (_nlos2_libdir, root, sl);
          memcpy (_nlos2_libdir + sl, LIBDIR, strlen (LIBDIR) + 1);
        }
      else
        _nlos2_libdir = LIBDIR;
    }

  _nlos2_localealiaspath = gnulocaledir;
  if (!_nlos2_localealiaspath)
    {
      if (root)
        {
          size_t sl = strlen (root);
          _nlos2_localealiaspath = (char *) malloc (sl + strlen (LOCALE_ALIAS_PATH) + 1);
          memcpy (_nlos2_localealiaspath, root, sl);
          memcpy (_nlos2_localealiaspath + sl, LOCALE_ALIAS_PATH, strlen (LOCALE_ALIAS_PATH) + 1);
        }
     else
        _nlos2_localealiaspath = LOCALE_ALIAS_PATH;
    }

  _nlos2_localedir = gnulocaledir;
  if (!_nlos2_localedir)
    {
      if (root)
        {
          size_t sl = strlen (root);
          _nlos2_localedir = (char *) malloc (sl + strlen (LOCALEDIR) + 1);
          memcpy (_nlos2_localedir, root, sl);
          memcpy (_nlos2_localedir + sl, LOCALEDIR, strlen (LOCALEDIR) + 1);
        }
      else
        _nlos2_localedir = LOCALEDIR;
    }

  if (strlen (_nlos2_localedir) <= MAXPATHLEN)
    strcpy (libintl_nl_default_dirname, _nlos2_localedir);
}
