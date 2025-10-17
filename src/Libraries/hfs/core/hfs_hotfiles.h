/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#ifndef __HFS_HOTFILES__
#define __HFS_HOTFILES__

#include <sys/appleapiopts.h>

#ifdef KERNEL
#ifdef __APPLE_API_PRIVATE


#define HFC_FILENAME	".hotfiles.btree"


/*
 * Temperature measurement constraints.
 */
#define HFC_DEFAULT_FILE_COUNT	 hfc_default_file_count
#define HFC_DEFAULT_DURATION     hfc_default_duration
#define HFC_CUMULATIVE_CYCLES	 3
#define HFC_MAXIMUM_FILE_COUNT	 hfc_max_file_count
#define HFC_MAXIMUM_FILESIZE	 hfc_max_file_size 
#define HFC_MINIMUM_TEMPERATURE  24


/*
 * Sync constraints.
 */
#define HFC_BLKSPERSYNC    300
#define HFC_FILESPERSYNC   50


/*
 * Hot file clustering stages.
 */
enum hfc_stage {
	HFC_DISABLED,
	HFC_IDLE,
	HFC_BUSY,
	HFC_RECORDING,
	HFC_EVALUATION,
	HFC_EVICTION,
	HFC_ADOPTION,
};


/* 
 * B-tree file key format (on-disk).
 */
struct HotFileKey {
	u_int16_t 	keyLength;	/* length of key, excluding this field */
	u_int8_t 	forkType;	/* 0 = data fork, FF = resource fork */
	u_int8_t 	pad;		/* make the other fields align on 32-bit boundary */
	u_int32_t 	temperature;	/* temperature recorded */
	u_int32_t 	fileID;		/* file ID */
};
typedef struct HotFileKey HotFileKey;

#define HFC_LOOKUPTAG   0xFFFFFFFF
#define HFC_KEYLENGTH	(sizeof(HotFileKey) - sizeof(u_int16_t))

/* 
 * B-tree header node user info (on-disk).
 */
struct HotFilesInfo {
	u_int32_t	magic;
	u_int32_t	version;
	u_int32_t	duration;    /* duration of sample period (secs) */
	u_int32_t	timebase;    /* start of recording period (GMT time in secs) */
	u_int32_t	timeleft;    /* time remaining in recording period (secs) */
	u_int32_t	threshold;
	u_int32_t	maxfileblks;
	union {
		u_int32_t	_maxfilecnt;   // on hdd's we track the max # of files
		u_int32_t	_usedblocks;   // on ssd's we track how many blocks are used
	} _u;
	u_int8_t	tag[32];
};

#define usedblocks _u._usedblocks
#define maxfilecnt _u._maxfilecnt

typedef struct HotFilesInfo HotFilesInfo;

#define HFC_MAGIC	0xFF28FF26
#define HFC_VERSION	1


struct hfsmount;
struct proc;
struct vnode;

/*
 * Hot File interface functions.
 */
int  hfs_hotfilesync (struct hfsmount *, vfs_context_t ctx);

int  hfs_recording_init(struct hfsmount *);
int  hfs_recording_suspend (struct hfsmount *);

int  hfs_addhotfile (struct vnode *);
int  hfs_removehotfile (struct vnode *);
int  hfs_hotfile_deleted(struct vnode *vp);   // called when a file is deleted
void hfs_repin_hotfiles(struct hfsmount *);

// call this to adjust the number of used hotfile blocks either up/down
int  hfs_hotfile_adjust_blocks(struct vnode *vp, int64_t num_blocks);

#endif /* __APPLE_API_PRIVATE */
#endif /* KERNEL */
#endif /* __HFS_HOTFILES__ */
