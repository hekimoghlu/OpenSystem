/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
 * NOTICE: This file was modified by SPARTA, Inc. in 2005 to introduce
 * support for mandatory and extensible security protections.  This notice
 * is included in support of clause 2.2 (b) of the Apple Public License,
 * Version 2.0.
 */
/*	$NetBSD: sem.h,v 1.5 1994/06/29 06:45:15 cgd Exp $	*/

/*
 * SVID compatible sem_internal.h file
 *
 * Author:  Daniel Boulet
 */
/*
 * John Bellardo modified the implementation for Darwin. 12/2000
 */

#ifndef _SYS_SEM__INTERNALH_
#define _SYS_SEM__INTERNALH_

#include <sys/sem.h>
#include <sys/cdefs.h>


/*
 * This structure is variant for 64 bits because of sem_otime and sem_ctime.
 */

#if __DARWIN_ALIGN_NATURAL
#pragma options align=natural
#endif

struct user_semid_ds {
	struct ipc_perm sem_perm;       /* [XSI] operation permission struct */
	struct sem      *sem_base;      /* 32 bit base ptr for semaphore set */
	unsigned short  sem_nsems;      /* [XSI] number of sems in set */
	user_time_t     sem_otime;      /* [XSI] last operation time */
	__int32_t       sem_pad1;       /* RESERVED: DO NOT USE! */
	user_time_t     sem_ctime;      /* [XSI] last change time */
	                                /* Times measured in secs since */
	                                /* 00:00:00 GMT, Jan. 1, 1970 */
	__int32_t       sem_pad2;       /* RESERVED: DO NOT USE! */
	__int32_t       sem_pad3[4];    /* RESERVED: DO NOT USE! */
};

#pragma pack(4)
struct user64_semid_ds {
	struct ipc_perm sem_perm;       /* [XSI] operation permission struct */
	int32_t sem_base;       /* 32 bit base ptr for semaphore set */
	unsigned short  sem_nsems;      /* [XSI] number of sems in set */
	user64_time_t   sem_otime;      /* [XSI] last operation time */
	int32_t sem_pad1;       /* RESERVED: DO NOT USE! */
	user64_time_t   sem_ctime;      /* [XSI] last change time */
	                                /* Times measured in secs since */
	                                /* 00:00:00 GMT, Jan. 1, 1970 */
	int32_t sem_pad2;       /* RESERVED: DO NOT USE! */
	int32_t sem_pad3[4];    /* RESERVED: DO NOT USE! */
};

struct user32_semid_ds {
	struct ipc_perm sem_perm;       /* [XSI] operation permission struct */
	int32_t sem_base;       /* 32 bit base ptr for semaphore set */
	unsigned short  sem_nsems;      /* [XSI] number of sems in set */
	user32_time_t   sem_otime;      /* [XSI] last operation time */
	int32_t sem_pad1;       /* RESERVED: DO NOT USE! */
	user32_time_t   sem_ctime;      /* [XSI] last change time */
	                                /* Times measured in secs since */
	                                /* 00:00:00 GMT, Jan. 1, 1970 */
	int32_t sem_pad2;       /* RESERVED: DO NOT USE! */
	int32_t sem_pad3[4];    /* RESERVED: DO NOT USE! */
};
#pragma pack()

union user_semun {
	user_addr_t     buf;            /* buffer for IPC_STAT & IPC_SET */
	user_addr_t     array;          /* array for GETALL & SETALL */
};
typedef union user_semun user_semun_t;

#if __DARWIN_ALIGN_NATURAL
#pragma options align=reset
#endif


/*
 * Kernel implementation stuff
 */
#define SEMVMX  32767           /* semaphore maximum value */
#define SEMAEM  16384           /* adjust on exit max value */

/*
 * Configuration parameters.  SEMMNI, SEMMNS, and SEMMNU are hard limits.
 * The code dynamically allocates enough memory to satisfy the current
 * demand in even increments of SEMMNI_INC, SEMMNS_INC, and SEMMNU_INC.
 * The code will never allocate more than the hard limits.  The *_INC's
 * are defined in the kernel section of the header.
 */
/*
 * Configuration parameters
 */
#ifndef SEMMNS                  /* # of semaphores in system */
#define SEMMNS  (1048576/sizeof(struct sem))
#endif                          /* no more than 1M of semaphore data */
#ifndef SEMMNI                  /* # of semaphore identifiers */
#define SEMMNI  SEMMNS          /* max of 1 for each semaphore */
#endif
#ifndef SEMUME
#define SEMUME  10              /* max # of undo entries per process */
#endif
#ifndef SEMMNU                  /* # of undo structures in system */
#define SEMMNU  SEMMNS          /* 1 for each semaphore.  This is quite large */
#endif                          /* This should be max 1 for each process */

/* shouldn't need tuning */
#ifndef SEMMAP
#define SEMMAP  30              /* # of entries in semaphore map */
#endif
#ifndef SEMMSL
#define SEMMSL  SEMMNS          /* max # of semaphores per id */
#endif
#ifndef SEMOPM
#define SEMOPM  5               /* max # of operations per semop call */
#endif


/*
 * Undo structure (internal: one per process)
 */
struct sem_undo {
	int     un_next_idx;    /* index of next active undo structure */
	struct  proc *un_proc;          /* owner of this structure */
	short   un_cnt;                 /* # of active entries */
	struct undo {
		short   une_adjval;     /* adjust on exit values */
		short   une_num;        /* semaphore # */
		int     une_id;         /* semid */
		struct undo *une_next;  /* next undo entry */
	} *un_ent;                      /* undo entries */
};

/*
 * semaphore info struct (internal; for administrative limits and ipcs)
 */
struct seminfo {
	int     semmap,         /* # of entries in semaphore map */
	    semmni,             /* # of semaphore identifiers */
	    semmns,             /* # of semaphores in system */
	    semmnu,             /* # of undo structures in system */
	    semmsl,             /* max # of semaphores per id */
	    semopm,             /* max # of operations per semop call */
	    semume,             /* max # of undo entries per process */
	    semusz,             /* size in bytes of undo structure */
	    semvmx,             /* semaphore maximum value */
	    semaem;             /* adjust on exit max value */
};
extern struct seminfo   seminfo;

/*
 * Kernel wrapper for the user-level structure
 */
struct semid_kernel {
	struct  user_semid_ds u;
	struct  label *label;   /* MAC framework label */
};


/* internal "mode" bits */
#define SEM_ALLOC       01000   /* semaphore is allocated */
#define SEM_DEST        02000   /* semaphore will be destroyed on last detach */

#define SEMMNI_INC      32      /* increment value for semaphore identifiers */
#define SEMMNS_INC      64      /* increment value for semaphores */
#define SEMMNU_INC      32      /* increment value for undo structures */

/*
 * Due to the way semaphore memory is allocated, we have to ensure that
 * SEMUSZ is properly aligned.
 *
 * We are not doing strange semaphore memory allocation anymore, so
 * these macros are no longer needed.
 */

/*
 * #define SEM_ALIGN(bytes) (((bytes) + (sizeof(long) - 1)) & ~(sizeof(long) - 1))
 */

/* actual size of an undo structure */
/*
 * #define SEMUSZ	SEM_ALIGN(offsetof(struct sem_undo, un_ent[SEMUME]))
 */
#define SEMUSZ          sizeof(struct sem_undo)

/*
 * Macro to find a particular sem_undo vector
 */
/* Until we can initialize seminfo.semusz to SEMUSZ, we hard code the size macro
 * in SEMU.  This should be fixed when (if) we implement dynamic pool sizes
 *
 * #define SEMU(ix)     ((struct sem_undo *)(((intptr_t)semu)+ix * seminfo.semusz))
 */
/*
 * This macro doesn't work because we are using a staticly allocated array
 * for semu now.
 * #define SEMU(ix)        ((struct sem_undo *)(((intptr_t)semu)+ix * SEMUSZ))
 */
#define SEMU(ix)        (&semu[ix])


/*
 * Process sem_undo vectors at proc exit.
 */
void    semexit(struct proc *p);

#endif /* !_SYS_SEM__INTERNALH_ */
