/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#include "uucp.h"

#include "sysdep.h"

#include <errno.h>

#if HAVE_LIMITS_H
#include <limits.h>
#endif

#if HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#if HAVE_OPENDIR
#if HAVE_DIRENT_H
#include <dirent.h>
#else /* ! HAVE_DIRENT_H */
#include <sys/dir.h>
#define dirent direct
#endif /* ! HAVE_DIRENT_H */
#endif /* HAVE_OPENDIR */

#if HAVE_FTW_H
#include <ftw.h>
#endif

#ifndef PATH_MAX
#ifdef MAXPATHLEN
#define PATH_MAX MAXPATHLEN
#else
#define PATH_MAX 1024
#endif
#endif

/* Traverse one level of a directory tree.  */

static int
ftw_dir (dirs, level, descriptors, dir, len, func)
     DIR **dirs;
     int level;
     int descriptors;
     char *dir;
     size_t len;
     int (*func) P((const char *file, const struct stat *status, int flag));
{
  int got;
  struct dirent *entry;

  got = 0;

  errno = 0;

  while ((entry = readdir (dirs[level])) != NULL)
    {
      size_t namlen;
      struct stat s;
      int flag, ret, newlev = 0;

      ++got;

      namlen = strlen (entry->d_name);
      if (entry->d_name[0] == '.'
	  && (namlen == 1 ||
	      (namlen == 2 && entry->d_name[1] == '.')))
	{
	  errno = 0;
	  continue;
	}

      if (namlen + len + 1 > PATH_MAX)
	{
#ifdef ENAMETOOLONG
	  errno = ENAMETOOLONG;
#else
	  errno = ENOMEM;
#endif
	  return -1;
	}

      dir[len] = '/';
      memcpy ((dir + len + 1), entry->d_name, namlen + 1);

      if (stat (dir, &s) < 0)
	{
	  if (errno != EACCES)
	    return -1;
	  flag = FTW_NS;
	}
      else if (S_ISDIR (s.st_mode))
	{
	  newlev = (level + 1) % descriptors;

	  if (dirs[newlev] != NULL)
	    closedir (dirs[newlev]);

	  dirs[newlev] = opendir (dir);
	  if (dirs[newlev] != NULL)
	    flag = FTW_D;
	  else
	    {
	      if (errno != EACCES)
		return -1;
	      flag = FTW_DNR;
	    }
	}
      else
	flag = FTW_F;

      ret = (*func) (dir, &s, flag);

      if (flag == FTW_D)
	{
	  if (ret == 0)
	    ret = ftw_dir (dirs, newlev, descriptors, dir,
			   namlen + len + 1, func);
	  if (dirs[newlev] != NULL)
	    {
	      int save;

	      save = errno;
	      closedir (dirs[newlev]);
	      errno = save;
	      dirs[newlev] = NULL;
	    }
	}

      if (ret != 0)
	return ret;

      if (dirs[level] == NULL)
	{
	  int skip;

	  dir[len] = '\0';
	  dirs[level] = opendir (dir);
	  if (dirs[level] == NULL)
	    return -1;
	  skip = got;
	  while (skip-- != 0)
	    {
	      errno = 0;
	      if (readdir (dirs[level]) == NULL)
		return errno == 0 ? 0 : -1;
	    }
	}

      errno = 0;
    }

  return errno == 0 ? 0 : -1;
}

/* Call a function on every element in a directory tree.  */

int
ftw (dir, func, descriptors)
     const char *dir;
     int (*func) P((const char *file, const struct stat *status, int flag));
     int descriptors;
{
  DIR **dirs;
  int c;
  DIR **p;
  size_t len;
  char buf[PATH_MAX + 1];
  struct stat s;
  int flag, ret;

  if (descriptors <= 0)
    descriptors = 1;

  dirs = (DIR **) malloc (descriptors * sizeof (DIR *));
  if (dirs == NULL)
    return -1;
  c = descriptors;
  p = dirs;
  while (c-- != 0)
    *p++ = NULL;

  len = strlen (dir);
  memcpy (buf, dir, len + 1);

  if (stat (dir, &s) < 0)
    {
      if (errno != EACCES)
	{
	  free ((pointer) dirs);
	  return -1;
	}
      flag = FTW_NS;
    }
  else if (S_ISDIR (s.st_mode))
    {
      dirs[0] = opendir (dir);
      if (dirs[0] != NULL)
	flag = FTW_D;
      else
	{
	  if (errno != EACCES)
	    {
	      free ((pointer) dirs);
	      return -1;
	    }
	  flag = FTW_DNR;
	}
    }
  else
    flag = FTW_F;

  ret = (*func) (buf, &s, flag);

  if (flag == FTW_D)
    {
      if (ret == 0)
	{
	  if (len == 1 && *buf == '/')
	    len = 0;
	  ret = ftw_dir (dirs, 0, descriptors, buf, len, func);
	}
      if (dirs[0] != NULL)
	{
	  int save;

	  save = errno;
	  closedir (dirs[0]);
	  errno = save;
	}
    }

  free ((pointer) dirs);
  return ret;
}
