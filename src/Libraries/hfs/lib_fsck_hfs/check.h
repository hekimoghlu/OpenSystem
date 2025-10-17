/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

//
//  check.h
//
//  Created by Adam Hijaze on 02/11/2022.
//

#ifndef check_hfs_h
#define check_hfs_h

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/disk.h>
#include <sys/stat.h>
#include <sys/mount.h>
#include <sys/ioctl.h>
#include <sys/sysctl.h>

#include "cache.h"
#include "lib_fsck_hfs.h"


/*
 * These definitions are duplicated from xnu's hfs_readwrite.c, and could live
 * in a shared header file if desired. On the other hand, the freeze and thaw
 * commands are not really supposed to be public.
 */
#ifndef    F_FREEZE_FS
#define F_FREEZE_FS     53              /* "freeze" all fs operations */
#define F_THAW_FS       54              /* "thaw" all fs operations */
#endif  // F_FREEZE_FS

#define     DIRTYEXIT       3   /* Filesystem Dirty, no checks */
#define     FIXEDROOTEXIT   4   /* Writeable Root Filesystem was fixed */
#define     EEXIT           8   /* Standard error exit. */
#define     MAJOREXIT       47  /* We had major errors when doing a early-exit verify */

typedef struct lib_fsck_ctx_t {
    fsck_client_ctx_t client;
    fsck_hfs_print_msg_type_funct_t print_msg_type;
    fsck_exfat_print_msg_num_func_t print_msg_num;
    fsck_ctx_t messages_context;
    fsck_exfat_print_debug_func_t print_debug;
} lib_fsck_ctx_t;

typedef struct fsck_state {
    int     fd;                 /* fd to the root-dir of the fs we're checking (only w/lfag == 1) */
    int     fsmodified;         /* 1 => write done to file system */
    int     fsreadfd;           /* file descriptor for reading file system */
    int     fswritefd;          /* file descriptor for writing file system */
    
    int     canWrite;           /* whether the target disk is writable ot not */
    char    hotroot;            /* checking root device */
    char    hotmount;           /* checking read-only mounted device */
    char    guiControl;         /* this app should output info for gui control */
    char    xmlControl;         /* output XML (plist) messages -- implies guiControl as well */
    char    rebuildBTree;       /* rebuild requested btree files */
    int     rebuildOptions;     /* pptions to indicate which btree should be rebuilt */
    char    modeSetting;        /* set the mode when creating "lost+found" directory */
    char    errorOnExit;        /* exit on first error */
    int     upgrading;          /* upgrading format */
    int     lostAndFoundMode;   /* octal mode used when creating "lost+found" directory */
    uint64_t reqCacheSize;      /* cache size requested by the caller (may be specified by the user via -c) */
    long        gBlockSize;     /* physical block size */
    int      detonatorRun;
    char *mountpoint;            /* device mount point */
    int     verbosityLevel;      /* verbosity level for printing debug messages */
    
    const char  *cdevname;          /* name of device being checked */
    char        *progname;
    char        lflag;              /* live fsck */
    char        nflag;              /* assume a no response */
    char        yflag;              /* assume a yes response */
    char        preen;              /* just fix normal inconsistencies */
    char        force;              /* force fsck even if clean (preen only) */
    char        quick;              /* quick check returns clean, dirty, or failure */
    char        debug;              /* output debugging info */
    char        disable_journal;    /* if debug, and set, do not simulate journal replay */
    char        scanflag;           /* scan entire disk for bad blocks */
    char        embedded;
    
    char        repLev;             /* repair level */
    char        chkLev;             /* check level */
    
    uint64_t    blockCount;         /* device blocks number */
    int         devBlockSize;       /* device block size */
    
    unsigned long cur_debug_level;  /* current debug level of fsck_hfs for printing debug messages */
} fsck_state_t;


fsck_state_t state;
lib_fsck_ctx_t ctx;


void fsck_print(lib_fsck_ctx_t c, LogMessageType type, const char *fmt, ...);
int fsckPrintFormat(lib_fsck_ctx_t c, int msgNum, ...);
void fsck_debug_print(lib_fsck_ctx_t c, int type, const char *fmt, ...);


/*
 * Routines from check.c
 */
void    start_progress(void);
void    draw_progress(int);
void    end_progress(void);
void    DumpData(const void *ptr, size_t sz, char *label);
int     CheckHFS(const char *rdevnode, int fsReadRef, int fsWriteRef,
                 int checkLevel, int repairLevel,
                 lib_fsck_ctx_t fsckContext,
                 int lostAndFoundMode, int canWrite,
                 int *modified, int liveMode, int rebuildOptions );

#endif /* check_hfs_h */
