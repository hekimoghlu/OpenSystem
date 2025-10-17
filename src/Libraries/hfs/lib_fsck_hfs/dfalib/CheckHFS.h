/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#ifndef __CheckHFS__
#define __CheckHFS__

/* External API to CheckHFS */

enum {
	kNeverCheck = 0,	/* never check (clean/dirty status only) */
	kDirtyCheck = 1,	/* only check if dirty */
	kAlwaysCheck = 2,	/* always check */
	kPartialCheck = 3,	/* used with kForceRepairs in order to set up environment */
	kForceCheck = 4,
	kMajorCheck = 5,	/* Check for major vs. minor errors */

	kNeverRepair = 0,	/* never repair */
	kMinorRepairs = 1,	/* only do minor repairs (fsck preen) */
	kMajorRepairs = 2,	/* do all possible repairs */
	kForceRepairs = 3,	/* force a repair of catalog B-Tree */
	
	kNeverLog = 0,
	kFatalLog = 1,		/* (fsck preen) */
	kVerboseLog = 2,	/* (Disk First Aid) */
	kDebugLog = 3
};

enum {
	R_NoMem			= 1,	/* not enough memory to do scavenge */
	R_IntErr		= 2,	/* internal Scavenger error */
	R_NoVol			= 3,	/* no volume in drive */
	R_RdErr			= 4,	/* unable to read from disk */
	R_WrErr			= 5,	/* unable to write to disk */
	R_BadSig		= 6,	/* not HFS/HFS+ signature */
	R_VFail			= 7,	/* verify failed */
	R_RFail			= 8,	/* repair failed */
	R_UInt			= 9,	/* user interrupt */
	R_Modified		= 10,	/* volume modifed by another app */
	R_BadVolumeHeader	= 11,	/* Invalid VolumeHeader */
	R_FileSharingIsON	= 12,	/* File Sharing is on */
	R_Dirty			= 13,	/* Dirty, but no checks were done */

	Max_RCode		= 13	/* maximum result code */
};

/* Option bits to indicate which type of btree to rebuild */
#define REBUILD_CATALOG		0x1
#define REBUILD_EXTENTS		0x2
#define REBUILD_ATTRIBUTE	0x4

extern int journal_replay(const char *);

#endif /* __CheckHFS__ */
