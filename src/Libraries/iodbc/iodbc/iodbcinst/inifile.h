/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#ifndef _INIFILE_H
#define _INIFILE_H

#include <fcntl.h>
#ifndef _MAC
#include <sys/types.h>
#endif

/* configuration file entry */
typedef struct TCFGENTRY
  {
    char *section;
    char *id;
    char *value;
    char *comment;
    unsigned short flags;
  }
TCFGENTRY, *PCFGENTRY;

/* values for flags */
#define CFE_MUST_FREE_SECTION	0x8000
#define CFE_MUST_FREE_ID	0x4000
#define CFE_MUST_FREE_VALUE	0x2000
#define CFE_MUST_FREE_COMMENT	0x1000

/* configuration file */
typedef struct TCFGDATA
  {
    char *fileName;		/* Current file name */

    int dirty;			/* Did we make modifications? */

    char *image;		/* In-memory copy of the file */
    size_t size;		/* Size of this copy (excl. \0) */
    time_t mtime;		/* Modification time */

    unsigned int numEntries;
    unsigned int maxEntries;
    PCFGENTRY entries;

    /* Compatibility */
    unsigned int cursor;
    char *section;
    char *id;
    char *value;
    char *comment;
    unsigned short flags;

  }
TCONFIG, *PCONFIG;

#define CFG_VALID		0x8000
#define CFG_EOF			0x4000

#define CFG_ERROR		0x0000
#define CFG_SECTION		0x0001
#define CFG_DEFINE		0x0002
#define CFG_CONTINUE		0x0003

#define CFG_TYPEMASK		0x000F
#define CFG_TYPE(X)		((X) & CFG_TYPEMASK)
#define _iodbcdm_cfg_valid(X)	((X) != NULL && ((X)->flags & CFG_VALID))
#define _iodbcdm_cfg_eof(X)	((X)->flags & CFG_EOF)
#define _iodbcdm_cfg_section(X)	(CFG_TYPE((X)->flags) == CFG_SECTION)
#define _iodbcdm_cfg_define(X)	(CFG_TYPE((X)->flags) == CFG_DEFINE)
#define _iodbcdm_cfg_cont(X)	(CFG_TYPE((X)->flags) == CFG_CONTINUE)

int _iodbcdm_cfg_init (PCONFIG * ppconf, const char *filename, int doCreate);
int _iodbcdm_cfg_done (PCONFIG pconfig);
int _iodbcdm_cfg_freeimage (PCONFIG pconfig);
int _iodbcdm_cfg_refresh (PCONFIG pconfig);
int _iodbcdm_cfg_storeentry (PCONFIG pconfig, char *section, char *id,
    char *value, char *comment, int dynamic);
int _iodbcdm_cfg_rewind (PCONFIG pconfig);
int _iodbcdm_cfg_nextentry (PCONFIG pconfig);
int _iodbcdm_cfg_find (PCONFIG pconfig, char *section, char *id);
int _iodbcdm_cfg_next_section (PCONFIG pconfig);

int _iodbcdm_cfg_write (PCONFIG pconfig, char *section, char *id, char *value);
int _iodbcdm_cfg_commit (PCONFIG pconfig);
int _iodbcdm_cfg_getstring (PCONFIG pconfig, char *section, char *id,
    char **valptr);
int _iodbcdm_cfg_getlong (PCONFIG pconfig, char *section, char *id,
    long *valptr);
int _iodbcdm_cfg_getshort (PCONFIG pconfig, char *section, char *id,
    short *valptr);
int _iodbcdm_cfg_search_init (PCONFIG * ppconf, const char *filename,
    int doCreate);
int _iodbcdm_list_entries (PCONFIG pCfg, LPCSTR lpszSection,
    LPSTR lpszRetBuffer, int cbRetBuffer);
int _iodbcdm_list_sections (PCONFIG pCfg, LPSTR lpszRetBuffer, int cbRetBuffer);
BOOL do_create_dsns (PCONFIG pCfg, PCONFIG pInfCfg, LPSTR szDriver,
    LPSTR szDSNS, LPSTR szDiz);
BOOL install_from_ini (PCONFIG pCfg, PCONFIG pOdbcCfg, LPSTR szInfFile,
    LPSTR szDriver, BOOL drivers);
int install_from_string (PCONFIG pCfg, PCONFIG pOdbcCfg, LPSTR lpszDriver,
    BOOL drivers);

#endif /* _INIFILE_H */
