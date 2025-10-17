/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
 * Copyright (C) 1995, 1996, 1997 Wolfgang Solfrank
 * Copyright (c) 1995 Martin Husemann
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by Martin Husemann
 *	and Wolfgang Solfrank.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef EXT_H
#define EXT_H

#include <sys/types.h>
#include <sys/stat.h>

#include "dosfs.h"
#include "lib_fsck_msdos.h"

#define	LOSTDIR	"LOST.DIR"

extern struct dosDirEntry *rootDir;

/*
 * function declarations
 */
int ask __P((int, const char *, ...));
int vask __P((fsck_client_ctx_t, int, const char *, va_list));

/*
 * Check filesystem given as arg
 */
typedef struct check_context_s {
    void *updater;
    void (*startPhase)(char *description, int64_t pendingUnits, int64_t totalCount, unsigned int *completedCount, void *updater);
    void (*endPhase)(char *description, void *updater);
    void *resource;
    size_t (*readHelper)(void *resource, void *buffer, size_t nbytes, off_t offset);
    size_t (*writeHelper)(void *resource, void *buffer, size_t nbytes, off_t offset);
    int (*fstatHelper)(void *resource, struct stat *);
    char *shadowPrefix;
    int shadowFD;
} check_context;
int checkfilesys(const char *fname, check_context *context);

/*
 * Return values of various functions
 */
#define	FSOK		0		/* Check was OK */
#define	FSBOOTMOD	1		/* Boot block was modified */
#define	FSDIRMOD	2		/* Some directory was modified */
#define	FSFATMOD	4		/* The FAT was modified */
#define	FSERROR		8		/* Some unrecovered error remains */
#define	FSFATAL		16		/* Some unrecoverable error occured */
#define FSDIRTY		32		/* File system is dirty */
#define FSFIXFAT	64		/* Fix file system FAT */

/*
 * read a boot block in a machine independend fashion and translate
 * it into our struct bootblock.
 */
int readboot __P((struct bootblock *, check_context*));

/*
 * Correct the FSInfo block.
 */
int writefsinfo __P((struct bootblock *, check_context*));

/*
 * Read a directory
 */
int resetDosDirSection(struct bootblock *boot, check_context *context);
void finishDosDirSection __P((void));
int handleDirTree(int, struct bootblock *boot, int, check_context *context);

/*
 * Small helper functions
 */
/*
 * Return the type of a reserved cluster as text
 */
char *rsrvdcltype __P((cl_t));

/*
 * Routines to read/write the FAT
 */
int fat_init(struct bootblock *boot, check_context *context);
void fat_uninit(void);
extern cl_t (*fat_get)(cl_t cluster, check_context *context);
extern int (*fat_set)(cl_t cluster, cl_t value, check_context *context);
int fat_free_unused(check_context *context);
int fat_flush(check_context *context);
/*
 * Determine whether a volume is dirty, without reading the entire FAT.
 */
int isdirty(struct bootblock *boot, int fat, check_context *context);
/*
 * Mark the volume "clean."
 */
int fat_mark_clean(check_context *context);


/*
 * Routines to track which clusters are in use (referenced by some directory entry).
 */
/* Returns error if memory cannot be allocated */
int initUseMap(struct bootblock *boot);
void freeUseMap(void);
/* Returns non-zero if block is marked allocated */
int isUsed(cl_t cluster);
/* Returns non-zero if cluster already marked allocated */
int markUsed(cl_t cluster);
/* Returns non-zero if cluster already marked free */
int markFree(cl_t cluster);

#endif
