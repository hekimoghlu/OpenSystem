/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
#include <iodbc.h>

#include <sql.h>
#include <sqlext.h>
#include <odbcinst.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <unicode.h>

#include <fcntl.h>
#include <sys/stat.h>

#include "herr.h"
#include "misc.h"
#include "iodbc_misc.h"

#ifdef _MAC
#include <getfpn.h>
#endif /* _MAC */


/*
 *  Parse a configuration from string (internal)
 */
int
_iodbcdm_cfg_parse_str_Internal (PCONFIG pconfig, char *str)
{
  char *s;
  int count;

  /* init image */
  _iodbcdm_cfg_freeimage (pconfig);
  if (str == NULL)
    {
      /* NULL string is ok */
      return 0;
    }
  s = pconfig->image = strdup (str);

  /* Add [ODBC] section */
  if (_iodbcdm_cfg_storeentry (pconfig, "ODBC", NULL, NULL, NULL, 0) == -1)
    return -1;

  for (count = 0; *s; count++)
    {
      char *keywd = NULL, *value;
      char *cp, *n;

      /* 
       *  Extract KEY=VALUE upto first ';'
       */
      for (cp = s; *cp && *cp != ';'; cp++)
	{
	  if (*cp == '{')
	    {
	      for (cp++; *cp && *cp != '}'; cp++)
		;
	    }
	}

      /*
       *  Store start of next token if available in n and terminate string
       */
      if (*cp)
	{
	  *cp = 0;
	  n = cp + 1;
	}
      else
	n = cp;

      /*
       *  Find '=' in string
       */
      for (cp = s; *cp && *cp != '='; cp++)
	;

      if (*cp)
	{
	  *cp++ = 0;
          keywd = s;
          value = cp;
	}
      else if (count == 0)
	{
	  /*
	   *  Handle missing DSN=... from the beginning of the string, e.g.:
	   *  'dsn_ora7;UID=scott;PWD=tiger'
	   */
          keywd = "DSN";
	  value = s;
	}

      if (keywd != NULL)
        {
          /* store entry */
          if (_iodbcdm_cfg_storeentry (pconfig, NULL,
		  keywd, value, NULL, 0) == -1)
            return -1;
	}

      /*
       *  Continue with next token
       */
      s = n;
    }

  /* we're done */
  pconfig->flags |= CFG_VALID;
  pconfig->dirty = 1;
  return 0;
}


/*
 *  Initialize a configuration from string
 */
int
_iodbcdm_cfg_init_str (PCONFIG *ppconf, void *str, int size, int wide)
{
  PCONFIG pconfig;

  *ppconf = NULL;

  /* init config */
  if ((pconfig = (PCONFIG) calloc (1, sizeof (TCONFIG))) == NULL)
    return -1;

  /* parse */
  if (_iodbcdm_cfg_parse_str (pconfig, str, size, wide) == -1)
    {
      _iodbcdm_cfg_done (pconfig);
      return -1;
    }

  /* we're done */
  *ppconf = pconfig;
  return 0;
}


/*
 *  Parse a configuration from string
 */
int
_iodbcdm_cfg_parse_str (PCONFIG pconfig, void *str, int size, int wide)
{
  int ret;
  char *_str;

  _str = wide ? (char *) dm_SQL_WtoU8 (str, size) : str;

  ret = _iodbcdm_cfg_parse_str_Internal (pconfig, _str);

  if (wide)
    MEM_FREE (_str);

  return ret;
}


#define CATBUF(buf, str, buf_sz)					\
  do {									\
    if (_iodbcdm_strlcat (buf, str, buf_sz) >= buf_sz)			\
      return -1;							\
  } while (0)

int
_iodbcdm_cfg_to_string (PCONFIG pconfig, char *section,
			char *buf, size_t buf_sz)
{
  BOOL atsection;

  if (_iodbcdm_cfg_rewind (pconfig) == -1)
    return -1;

  atsection = FALSE;
  buf[0] = '\0';
  while (_iodbcdm_cfg_nextentry (pconfig) == 0)
    {
      if (atsection)
        {
          if (_iodbcdm_cfg_section (pconfig))
            break;
          else if (_iodbcdm_cfg_define (pconfig))
            {
              if (buf[0] != '\0')
                CATBUF (buf, ";", buf_sz);
              CATBUF (buf, pconfig->id, buf_sz);
              CATBUF (buf, "=", buf_sz);
              CATBUF (buf, pconfig->value, buf_sz);
            }
        }
      else if (_iodbcdm_cfg_section (pconfig) &&
	       !strcasecmp (pconfig->section, section))
        atsection = TRUE;
    }
  return 0;
}
