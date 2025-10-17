/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include "archive_platform.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "archive.h"
#include "archive_private.h"

/* A table that maps names to functions. */
static const
struct { const char *name; int (*format)(struct archive *); int (*filter)(struct archive *);  } names[] =
{
	{ ".7z",	archive_write_set_format_7zip,            archive_write_add_filter_none},
	{ ".zip",	archive_write_set_format_zip,             archive_write_add_filter_none},
	{ ".jar",	archive_write_set_format_zip,             archive_write_add_filter_none},
	{ ".cpio",	archive_write_set_format_cpio,            archive_write_add_filter_none},
	{ ".iso",	archive_write_set_format_iso9660,         archive_write_add_filter_none},
#if defined(__FreeBSD__) || defined(__FreeBSD_kernel__) || defined(__NetBSD__) || defined(__OpenBSD__)
	{ ".a",	        archive_write_set_format_ar_bsd,          archive_write_add_filter_none},
	{ ".ar",	archive_write_set_format_ar_bsd,          archive_write_add_filter_none},
#else         
	{ ".a",	        archive_write_set_format_ar_svr4,         archive_write_add_filter_none},
	{ ".ar",	archive_write_set_format_ar_svr4,         archive_write_add_filter_none},
#endif        
	{ ".tar",	archive_write_set_format_pax_restricted,  archive_write_add_filter_none},
	{ ".tgz",	archive_write_set_format_pax_restricted,  archive_write_add_filter_gzip},
	{ ".tar.gz",	archive_write_set_format_pax_restricted,  archive_write_add_filter_gzip},
	{ ".tar.bz2",	archive_write_set_format_pax_restricted,  archive_write_add_filter_bzip2},
	{ ".tar.xz",	archive_write_set_format_pax_restricted,  archive_write_add_filter_xz},
	{ NULL,		NULL,                             NULL }
};

static 
int cmpsuff(const char *str, const char *suffix)
{
  size_t length_str, length_suffix;

  if ((str == NULL) || (suffix == NULL))
    return -1;

  length_str = strlen(str);
  length_suffix = strlen(suffix);

  if (length_str >= length_suffix) {
    return strcmp(str + (length_str - length_suffix), suffix);
  } else {
    return -1;
  }
}

static int get_array_index(const char *name)
{
  int i;

  for (i = 0; names[i].name != NULL; i++) 
  {
    if (cmpsuff(name, names[i].name) == 0)
      return i;
  }    
  return -1;
  
}

int
archive_write_set_format_filter_by_ext(struct archive *a, const char *filename)
{
  int names_index = get_array_index(filename);
  
  if (names_index >= 0)
  {  
    int format_state = (names[names_index].format)(a);
    if (format_state == ARCHIVE_OK)
      return ((names[names_index].filter)(a));
    else
      return format_state;
  }    

  archive_set_error(a, EINVAL, "No such format '%s'", filename);
  a->state = ARCHIVE_STATE_FATAL;
  return (ARCHIVE_FATAL);
}

int
archive_write_set_format_filter_by_ext_def(struct archive *a, const char *filename, const char * def_ext)
{
  int names_index = get_array_index(filename);
  
  if (names_index < 0)
    names_index = get_array_index(def_ext);
  
  if (names_index >= 0)
  {  
    int format_state = (names[names_index].format)(a);
    if (format_state == ARCHIVE_OK)
      return ((names[names_index].filter)(a));
    else
      return format_state;
  }    

  archive_set_error(a, EINVAL, "No such format '%s'", filename);
  a->state = ARCHIVE_STATE_FATAL;
  return (ARCHIVE_FATAL);
}




