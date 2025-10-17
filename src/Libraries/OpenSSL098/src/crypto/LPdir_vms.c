/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <descrip.h>
#include <namdef.h>
#include <rmsdef.h>
#include <libfildef.h>
#include <lib$routines.h>
#include <strdef.h>
#include <str$routines.h>
#include <stsdef.h>
#ifndef LPDIR_H
#include "LPdir.h"
#endif

/* Because some compiler options hide this macor */
#ifndef EVMSERR
#define EVMSERR		65535  /* error for non-translatable VMS errors */
#endif

struct LP_dir_context_st
{
  unsigned long VMS_context;
#ifdef NAML$C_MAXRSS
  char filespec[NAML$C_MAXRSS+1];
  char result[NAML$C_MAXRSS+1];
#else
  char filespec[256];
  char result[256];
#endif
  struct dsc$descriptor_d filespec_dsc;
  struct dsc$descriptor_d result_dsc;
};

const char *LP_find_file(LP_DIR_CTX **ctx, const char *directory)
{
  int status;
  char *p, *r;
  size_t l;
  unsigned long flags = 0;
#ifdef NAML$C_MAXRSS
  flags |= LIB$M_FIL_LONG_NAMES;
#endif

  if (ctx == NULL || directory == NULL)
    {
      errno = EINVAL;
      return 0;
    }

  errno = 0;
  if (*ctx == NULL)
    {
      size_t filespeclen = strlen(directory);
      char *filespec = NULL;

      if (filespeclen == 0)
	{
	  errno = ENOENT;
	  return 0;
	}

      /* MUST be a VMS directory specification!  Let's estimate if it is. */
      if (directory[filespeclen-1] != ']'
	  && directory[filespeclen-1] != '>'
	  && directory[filespeclen-1] != ':')
	{
	  errno = EINVAL;
	  return 0;
	}

      filespeclen += 4;		/* "*.*;" */

      if (filespeclen >
#ifdef NAML$C_MAXRSS
	  NAML$C_MAXRSS
#else
	  255
#endif
	  )
	{
	  errno = ENAMETOOLONG;
	  return 0;
	}

      *ctx = (LP_DIR_CTX *)malloc(sizeof(LP_DIR_CTX));
      if (*ctx == NULL)
	{
	  errno = ENOMEM;
	  return 0;
	}
      memset(*ctx, '\0', sizeof(LP_DIR_CTX));

      strcpy((*ctx)->filespec,directory);
      strcat((*ctx)->filespec,"*.*;");
      (*ctx)->filespec_dsc.dsc$w_length = filespeclen;
      (*ctx)->filespec_dsc.dsc$b_dtype = DSC$K_DTYPE_T;
      (*ctx)->filespec_dsc.dsc$b_class = DSC$K_CLASS_S;
      (*ctx)->filespec_dsc.dsc$a_pointer = (*ctx)->filespec;
      (*ctx)->result_dsc.dsc$w_length = 0;
      (*ctx)->result_dsc.dsc$b_dtype = DSC$K_DTYPE_T;
      (*ctx)->result_dsc.dsc$b_class = DSC$K_CLASS_D;
      (*ctx)->result_dsc.dsc$a_pointer = 0;
    }

  (*ctx)->result_dsc.dsc$w_length = 0;
  (*ctx)->result_dsc.dsc$b_dtype = DSC$K_DTYPE_T;
  (*ctx)->result_dsc.dsc$b_class = DSC$K_CLASS_D;
  (*ctx)->result_dsc.dsc$a_pointer = 0;

  status = lib$find_file(&(*ctx)->filespec_dsc, &(*ctx)->result_dsc,
			 &(*ctx)->VMS_context, 0, 0, 0, &flags);

  if (status == RMS$_NMF)
    {
      errno = 0;
      vaxc$errno = status;
      return NULL;
    }

  if(!$VMS_STATUS_SUCCESS(status))
    {
      errno = EVMSERR;
      vaxc$errno = status;
      return NULL;
    }

  /* Quick, cheap and dirty way to discard any device and directory,
     since we only want file names */
  l = (*ctx)->result_dsc.dsc$w_length;
  p = (*ctx)->result_dsc.dsc$a_pointer;
  r = p;
  for (; *p; p++)
    {
      if (*p == '^' && p[1] != '\0') /* Take care of ODS-5 escapes */
	{
	  p++;
	}
      else if (*p == ':' || *p == '>' || *p == ']')
	{
	  l -= p + 1 - r;
	  r = p + 1;
	}
      else if (*p == ';')
	{
	  l = p - r;
	  break;
	}
    }

  strncpy((*ctx)->result, r, l);
  (*ctx)->result[l] = '\0';
  str$free1_dx(&(*ctx)->result_dsc);

  return (*ctx)->result;
}

int LP_find_file_end(LP_DIR_CTX **ctx)
{
  if (ctx != NULL && *ctx != NULL)
    {
      int status = lib$find_file_end(&(*ctx)->VMS_context);

      free(*ctx);

      if(!$VMS_STATUS_SUCCESS(status))
	{
	  errno = EVMSERR;
	  vaxc$errno = status;
	  return 0;
	}
      return 1;
    }
  errno = EINVAL;
  return 0;
}

