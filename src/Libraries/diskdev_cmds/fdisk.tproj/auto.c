/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
/*
 * Auto partitioning code.
 */

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <err.h>
#include "disk.h"
#include "mbr.h"
#include "auto.h"

int AUTO_boothfs __P((disk_t *, mbr_t *));
int AUTO_hfs __P((disk_t *, mbr_t *));
int AUTO_dos __P((disk_t *, mbr_t *));
int AUTO_raid __P((disk_t *, mbr_t *));

/* The default style is the first one in the list */
struct _auto_style {
  char *style_name;
  int (*style_fn)(disk_t *, mbr_t *);
  char *description;
} style_fns[] = {
  {"boothfs", AUTO_boothfs, "8Mb boot plus HFS+ root partition"},
  {"hfs",     AUTO_hfs,     "Entire disk as one HFS+ partition"},
  {"dos",     AUTO_dos,     "Entire disk as one DOS partition"},
  {"raid",    AUTO_raid,    "Entire disk as one 0xAC partition"},
  {0,0}
};

void
AUTO_print_styles(FILE *f)
{
  struct _auto_style *fp;
  int i;

  for (i=0, fp = &style_fns[0]; fp->style_name != NULL; i++, fp++) {
    fprintf(f, "  %-10s  %s%s\n", fp->style_name, fp->description, (i==0) ? " (default)" : "");
  }
}


int
AUTO_init(disk_t *disk, char *style, mbr_t *mbr)
{
  struct _auto_style *fp;

  for (fp = &style_fns[0]; fp->style_name != NULL; fp++) {
    /* If style is NULL, use the first (default) style */
    if (style == NULL || strcasecmp(style, fp->style_name) == 0) {
      return (*fp->style_fn)(disk, mbr);
    }
  }
  warnx("No such auto-partition style %s", style);
  return AUTO_ERR;
}


static int
use_whole_disk(disk_t *disk, unsigned char id, mbr_t *mbr)
{
  MBR_clear(mbr);
  mbr->part[0].id = id;
  mbr->part[0].bs = 63;
  mbr->part[0].ns = disk->real->size - 63;
  PRT_fix_CHS(disk, &mbr->part[0], 0);
  return AUTO_OK;
}

/* DOS style: one partition for the whole disk */
int
AUTO_dos(disk_t *disk, mbr_t *mbr)
{
  int cc;
  cc = use_whole_disk(disk, 0x0C, mbr);
  if (cc == AUTO_OK) {
    mbr->part[0].flag = DOSACTIVE;
  }
  return cc;
}

/* HFS style: one partition for the whole disk */
int
AUTO_hfs(disk_t *disk, mbr_t *mbr)
{
  int cc;
  cc = use_whole_disk(disk, 0xAF, mbr);
  if (cc == AUTO_OK) {
    mbr->part[0].flag = DOSACTIVE;
  }
  return cc;
}

/* One boot partition, one HFS+ root partition */
int
AUTO_boothfs (disk_t *disk, mbr_t *mbr)
{
  /* Check disk size. */
  if (disk->real->size < 16 * 2048) {
    errx(1, "Disk size must be greater than 16Mb");
    return AUTO_ERR;
  }

  MBR_clear(mbr);

  /* 8MB boot partition */
  mbr->part[0].id = 0xAB;
  mbr->part[0].bs = 63;
  mbr->part[0].ns = 8 * 1024 * 2;
  mbr->part[0].flag = DOSACTIVE;
  PRT_fix_CHS(disk, &mbr->part[0], 0);

  /* Rest of the disk for rooting */
  mbr->part[1].id = 0xAF;
  mbr->part[1].bs = (mbr->part[0].bs + mbr->part[0].ns);
  mbr->part[1].ns = disk->real->size - mbr->part[0].ns - 63;
  PRT_fix_CHS(disk, &mbr->part[1], 1);

  return AUTO_OK;
}



/* RAID style: one 0xAC partition for the whole disk */
int
AUTO_raid(disk_t *disk, mbr_t *mbr)
{
  return use_whole_disk(disk, 0xAC, mbr);
}

