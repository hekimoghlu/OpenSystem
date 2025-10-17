/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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
#include "uucnfi.h"

#if USE_RCS_ID
const char _uuconf_cnfnms_rcsid[] = "$Id: cnfnms.c,v 1.2 2002/03/05 19:10:42 ian Rel $";
#endif

#include <errno.h>

int
uuconf_config_files (pglobal, qnames)
     pointer pglobal;
     struct uuconf_config_file_names* qnames;
{
  struct sglobal *qglobal = (struct sglobal *) pglobal;

  qnames->uuconf_ztaylor_config = qglobal->qprocess->zconfigfile;
  qnames->uuconf_pztaylor_sys =
    (const char * const *) qglobal->qprocess->pzsysfiles;
  qnames->uuconf_pztaylor_port =
    (const char * const *) qglobal->qprocess->pzportfiles;
  qnames->uuconf_pztaylor_dial =
    (const char * const *) qglobal->qprocess->pzdialfiles;
  qnames->uuconf_pzdialcode =
    (const char * const *) qglobal->qprocess->pzdialcodefiles;
  qnames->uuconf_pztaylor_pwd =
    (const char * const *) qglobal->qprocess->pzpwdfiles;
  qnames->uuconf_pztaylor_call =
    (const char * const *) qglobal->qprocess->pzcallfiles;

  qnames->uuconf_zv2_systems = qglobal->qprocess->zv2systems;
  qnames->uuconf_zv2_device = qglobal->qprocess->zv2devices;
  qnames->uuconf_zv2_userfile = qglobal->qprocess->zv2userfile;
  qnames->uuconf_zv2_cmds = qglobal->qprocess->zv2cmds;

  qnames->uuconf_pzhdb_systems =
    (const char * const *) qglobal->qprocess->pzhdb_systems;
  qnames->uuconf_pzhdb_devices =
    (const char * const *) qglobal->qprocess->pzhdb_devices;
  qnames->uuconf_pzhdb_dialers =
    (const char * const *) qglobal->qprocess->pzhdb_dialers;

  qnames->uuconf_zhdb_permissions = NULL;
#if HAVE_HDB_CONFIG
  if (qglobal->qprocess->fhdb)
    {    
      /* FIXME: There is a memory leak here.  */
      qnames->uuconf_zhdb_permissions =
	(char *) uuconf_malloc(qglobal->pblock,
			       (sizeof OLDCONFIGLIB
				+ sizeof HDB_PERMISSIONS - 1));
      if (qnames->uuconf_zhdb_permissions == NULL)
	{
	  qglobal->ierrno = errno;
	  return UUCONF_MALLOC_FAILED | UUCONF_ERROR_ERRNO;
	}
      memcpy((pointer) qnames->uuconf_zhdb_permissions, (pointer) OLDCONFIGLIB,
	     sizeof OLDCONFIGLIB - 1);
      memcpy((pointer) (qnames->uuconf_zhdb_permissions
			+ sizeof OLDCONFIGLIB
			- 1),
	     (pointer) HDB_PERMISSIONS, sizeof HDB_PERMISSIONS);
    }
#endif

  return UUCONF_SUCCESS;
}
