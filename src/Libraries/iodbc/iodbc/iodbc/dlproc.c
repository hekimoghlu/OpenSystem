/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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

#include <dlproc.h>

#include <herr.h>
#include <henv.h>
#include <hdbc.h>

#include <itrace.h>

char *odbcapi_symtab[] =
{
    "UNKNOWN FUNCTION"
#define FUNCDEF(A, B, C)	,C
#include "henv.ci"
#undef FUNCDEF
};


HPROC
_iodbcdm_getproc (HDBC hdbc, int idx)
{
  CONN (pdbc, hdbc);
  ENV_t *penv;
  HPROC *phproc;

  if (idx <= 0 || idx >= __LAST_API_FUNCTION__)
    return SQL_NULL_HPROC;

  penv = (ENV_t *) (pdbc->henv);

  if (penv == NULL)
    return SQL_NULL_HPROC;

  phproc = penv->dllproc_tab + idx;

  if (*phproc == SQL_NULL_HPROC)
    *phproc = _iodbcdm_dllproc (penv->hdll, odbcapi_symtab[idx]);

  return *phproc;
}


static dlproc_t *pRoot = NULL;


HDLL
_iodbcdm_dllopen (char *path)
{
  dlproc_t *pDrv = NULL, *p;

  /*
   *  Check if we have already loaded the driver
   */
  for (p = pRoot; p; p = p->next)
    {
      if (STREQ (p->path, path))
	{
	  pDrv = p;
	  break;
	}
    }

  /*
   *  If already loaded, increase ref counter
   */
  if (pDrv)
    {
      pDrv->refcount++;

      /*
       *  If the driver was unloaded, load it again
       */
      if (pDrv->dll == NULL)
	pDrv->dll = (HDLL) DLL_OPEN (path);

      return pDrv->dll;
    }

  /*
   *  Initialize new structure
   */
  if ((pDrv = calloc (1, sizeof (dlproc_t))) == NULL)
    return NULL;

  pDrv->refcount = 1;
  pDrv->path = STRDUP (path);
  pDrv->dll = (HDLL) DLL_OPEN (path);

  /*
   *  Add to linked list
   */
  pDrv->next = pRoot;
  pRoot = pDrv;

  return pDrv->dll;
}


HPROC
_iodbcdm_dllproc (HDLL hdll, char *sym)
{
  return (HPROC) DLL_PROC (hdll, sym);
}


int
_iodbcdm_dllclose (HDLL hdll)
{
  dlproc_t *pDrv = NULL, *p;

  /*
   *  Find loaded driver
   */
  for (p = pRoot; p; p = p->next)
    {
      if (p->dll == hdll)
	{
	  pDrv = p;
	  break;
	}
    }

  /*
   *  Not found
   */
  if (!pDrv)
    return -1;

  /*
   *  Decrease reference counter
   */
  pDrv->refcount--;

  /*
   *  Check if it is possible to unload the driver safely
   * 
   *  NOTE: Some drivers set explicit on_exit hooks, which makes it
   *        impossible for the driver manager to unload the driver
   *        as this would crash the executable at exit.
   */
  if (pDrv->refcount == 0 && pDrv->safe_unload)
    {
      DLL_CLOSE (pDrv->dll);
      pDrv->dll = NULL;
    }

  return 0;
}


char *
_iodbcdm_dllerror ()
{
  return DLL_ERROR ();
}


/* 
 *  If driver manager determines this driver is safe, flag the driver can
 *  be unloaded if not used.
 */
void
_iodbcdm_safe_unload (HDLL hdll)
{
  dlproc_t *pDrv = NULL, *p;

  /*
   *  Find loaded driver
   */
  for (p = pRoot; p; p = p->next)
    {
      if (p->dll == hdll)
	{
	  pDrv = p;
	  break;
	}
    }

  /*
   *  Driver not found
   */
  if (!pDrv)
    return;

  pDrv->safe_unload = 1;
}
