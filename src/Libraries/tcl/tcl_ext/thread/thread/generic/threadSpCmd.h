/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#ifndef _SP_H_
#define _SP_H_

#include <tcl.h>

/*
 * The following structure defines a locking bucket. A locking
 * bucket is associated with a mutex and protects access to 
 * objects stored in bucket hash table.
 */

typedef struct SpBucket {
    Tcl_Mutex lock;            /* For locking the bucket */
    Tcl_Condition cond;        /* For waiting on threads to release items */
    Tcl_ThreadId lockt;        /* Thread holding the lock */
    Tcl_HashTable handles;     /* Hash table of given-out handles in bucket */
    struct Container *freeCt;  /* List of free Tcl-object containers */
} SpBucket;

#define NUMSPBUCKETS 32

/*
 * All types of mutexes share this common part.
 */

typedef struct Sp_AnyMutex_ {
    int lockcount;              /* If !=0 mutex is locked */
    int numlocks;               /* Number of times the mutex got locked */
    Tcl_Mutex lock;             /* Regular mutex */
    Tcl_ThreadId owner;         /* Current lock owner thread (-1 = any) */
} Sp_AnyMutex;

/*
 * Implementation of the exclusive mutex.
 */

typedef struct Sp_ExclusiveMutex_ {
    int lockcount;              /* Flag: 1-locked, 0-not locked */
    int numlocks;               /* Number of times the mutex got locked */
    Tcl_Mutex lock;             /* Regular mutex */
    Tcl_ThreadId owner;         /* Current lock owner thread */
    /* --- */
    Tcl_Mutex mutex;            /* Mutex being locked */
} Sp_ExclusiveMutex_;

typedef Sp_ExclusiveMutex_* Sp_ExclusiveMutex;

/*
 * Implementation of the recursive mutex.
 */

typedef struct Sp_RecursiveMutex_ {
    int lockcount;              /* # of times this mutex is locked */
    int numlocks;               /* Number of time the mutex got locked */
    Tcl_Mutex lock;             /* Regular mutex */
    Tcl_ThreadId owner;         /* Current lock owner thread */
    /* --- */
    Tcl_Condition cond;         /* Wait to be allowed to lock the mutex */
} Sp_RecursiveMutex_;

typedef Sp_RecursiveMutex_* Sp_RecursiveMutex;

/*
 * Implementation of the read/writer mutex.
 */

typedef struct Sp_ReadWriteMutex_ {
    int lockcount;              /* >0: # of readers, -1: sole writer */
    int numlocks;               /* Number of time the mutex got locked */
    Tcl_Mutex lock;             /* Regular mutex */
    Tcl_ThreadId owner;         /* Current lock owner thread */
    /* --- */
    unsigned int numrd;	        /* # of readers waiting for lock */
    unsigned int numwr;         /* # of writers waiting for lock */
    Tcl_Condition rcond;        /* Reader lockers wait here */
    Tcl_Condition wcond;        /* Writer lockers wait here */
} Sp_ReadWriteMutex_;

typedef Sp_ReadWriteMutex_* Sp_ReadWriteMutex;


/*
 * API for exclusive mutexes.
 */

int  Sp_ExclusiveMutexLock(Sp_ExclusiveMutex *mutexPtr);
int  Sp_ExclusiveMutexIsLocked(Sp_ExclusiveMutex *mutexPtr);
int  Sp_ExclusiveMutexUnlock(Sp_ExclusiveMutex *mutexPtr);
void Sp_ExclusiveMutexFinalize(Sp_ExclusiveMutex *mutexPtr);

/*
 * API for recursive mutexes.
 */

int  Sp_RecursiveMutexLock(Sp_RecursiveMutex *mutexPtr);
int  Sp_RecursiveMutexIsLocked(Sp_RecursiveMutex *mutexPtr);
int  Sp_RecursiveMutexUnlock(Sp_RecursiveMutex *mutexPtr);
void Sp_RecursiveMutexFinalize(Sp_RecursiveMutex *mutexPtr);

/*
 * API for reader/writer mutexes.
 */

int  Sp_ReadWriteMutexRLock(Sp_ReadWriteMutex *mutexPtr);
int  Sp_ReadWriteMutexWLock(Sp_ReadWriteMutex *mutexPtr);
int  Sp_ReadWriteMutexIsLocked(Sp_ReadWriteMutex *mutexPtr);
int  Sp_ReadWriteMutexUnlock(Sp_ReadWriteMutex *mutexPtr);
void Sp_ReadWriteMutexFinalize(Sp_ReadWriteMutex *mutexPtr);

#endif /* _SP_H_ */

/* EOF $RCSfile: threadSpCmd.h,v $ */

/* Emacs Setup Variables */
/* Local Variables:      */
/* mode: C               */
/* indent-tabs-mode: nil */
/* c-basic-offset: 4     */
/* End:                  */
