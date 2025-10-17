/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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

#ifdef HAVE_PWD_H
#  undef __NO_NET_API /* required for building for AmigaOS */
#  include <pwd.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#include <curl/mprintf.h>

#include "tool_findfile.h"

#include "memdebug.h" /* keep this as LAST include */

struct finder {
  const char *env;
  const char *append;
  bool withoutdot;
};

/* The order of the variables below is important, as the index number is used
   in the findfile() function */
static const struct finder conf_list[] = {
  { "CURL_HOME", NULL, FALSE },
  { "XDG_CONFIG_HOME", NULL, FALSE }, /* index == 1, used in the code */
  { "HOME", NULL, FALSE },
#ifdef _WIN32
  { "USERPROFILE", NULL, FALSE },
  { "APPDATA", NULL, FALSE },
  { "USERPROFILE", "\\Application Data", FALSE},
#endif
  /* these are for .curlrc if XDG_CONFIG_HOME is not defined */
  { "CURL_HOME", "/.config", TRUE },
  { "HOME", "/.config", TRUE },

  { NULL, NULL, FALSE }
};

static char *checkhome(const char *home, const char *fname, bool dotscore)
{
  const char pref[2] = { '.', '_' };
  int i;
  for(i = 0; i < (dotscore ? 2 : 1); i++) {
    char *c;
    if(dotscore)
      c = curl_maprintf("%s" DIR_CHAR "%c%s", home, pref[i], &fname[1]);
    else
      c = curl_maprintf("%s" DIR_CHAR "%s", home, fname);
    if(c) {
      int fd = open(c, O_RDONLY);
      if(fd >= 0) {
        char *path = strdup(c);
        close(fd);
        curl_free(c);
        return path;
      }
      curl_free(c);
    }
  }
  return NULL;
}

/*
 * findfile() - return the full path name of the file.
 *
 * If 'dotscore' is TRUE, then check for the file first with a leading dot
 * and then with a leading underscore.
 *
 * 1. Iterate over the environment variables in order, and if set, check for
 *    the given file to be accessed there, then it is a match.
 * 2. Non-windows: try getpwuid
 */
char *findfile(const char *fname, int dotscore)
{
  int i;
  bool xdg = FALSE;
  DEBUGASSERT(fname && fname[0]);
  DEBUGASSERT((dotscore != 1) || (fname[0] == '.'));

  if(!fname[0])
    return NULL;

  for(i = 0; conf_list[i].env; i++) {
    char *home = curl_getenv(conf_list[i].env);
    if(home) {
      char *path;
      const char *filename = fname;
      if(i == 1 /* XDG_CONFIG_HOME */)
        xdg = TRUE;
      if(!home[0]) {
        curl_free(home);
        continue;
      }
      if(conf_list[i].append) {
        char *c = curl_maprintf("%s%s", home, conf_list[i].append);
        curl_free(home);
        if(!c)
          return NULL;
        home = c;
      }
      if(conf_list[i].withoutdot) {
        if(!dotscore || xdg) {
          /* this is not looking for .curlrc, or the XDG_CONFIG_HOME was
             defined so we skip the extended check */
          curl_free(home);
          continue;
        }
        filename++; /* move past the leading dot */
        dotscore = 0; /* disable it for this check */
      }
      path = checkhome(home, filename, dotscore ? dotscore - 1 : 0);
      curl_free(home);
      if(path)
        return path;
    }
  }
#if defined(HAVE_GETPWUID) && defined(HAVE_GETEUID)
  {
    struct passwd *pw = getpwuid(geteuid());
    if(pw) {
      char *home = pw->pw_dir;
      if(home && home[0])
        return checkhome(home, fname, FALSE);
    }
  }
#endif /* PWD-stuff */
  return NULL;
}
