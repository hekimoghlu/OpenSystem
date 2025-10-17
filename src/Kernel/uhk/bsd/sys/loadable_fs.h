/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
/*	@(#)loadable_fs.h	2.0	26/06/90	(c) 1990 NeXT	*/

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 *
 * W I L L    D R A S T I C A L L Y   C H A N G E   S O O N
 * U S E   A T  Y O U R   O W N  R I S K
 *
 * XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */


/*
 * loadable_fs.h - message struct for loading and initializing loadable
 *		   file systems.
 */

#ifndef _SYS_LOADABLE_FS_
#define _SYS_LOADABLE_FS_


/*
 * Constants for Loadable FS Utilities (in "/System/Library/Filesystems")
 *
 * Example of a /usr/filesystems directory
 *
 * /usr/filesystems/dos.fs/dos.util		utility with which WSM
 *							communicates
 * /usr/filesystems/dos.fs/dos.name             "DOS Floppy"
 * /usr/filesystems/dos.fs/dos_reloc		actual loadable filesystem
 * /usr/filesystems/dos.fs/dos.openfs.tiff	"open folder" icon
 * /usr/filesystems/dos.fs/dos.fs.tiff		"closed folder" icon
 */
#define FS_DIR_LOCATION         "/System/Library/Filesystems"
#define FS_DIR_SUFFIX           ".fs"
#define FS_UTIL_SUFFIX          ".util"

/*
 * .util program commands - all sent in the form "-p" or "-m" ... as argv[1].
 */
#define FSUC_PROBE              'p'     /* probe FS for mount or init */
/* example usage: foo.util -p fd0 removable writable */

#define FSUC_PROBEFORINIT       'P'     /* probe FS for init only */
/* example usage: foo.util -P fd0 removable */

#define FSUC_MOUNT              'm'     /* mount FS */
/* example usage: foo.util -m fd0 /bar removable writable */

#define FSUC_REPAIR             'r'     /* repair ('fsck') FS */
/* example usage: foo.util -r fd0 removable */

#define FSUC_UNMOUNT            'u'     /* unmount FS */
/* example usage: foo.util -u fd0 /bar */

/* The following is not used by Workspace Manager */
#define FSUC_MOUNT_FORCE        'M'     /* like FSUC_MOUNT, but proceed even on
	                                 * error. */
/*
 * Return codes from .util program
 */
#define FSUR_RECOGNIZED         (-1)    /* response to FSUC_PROBE; implies that
	                                 * a mount is possible */
#define FSUR_UNRECOGNIZED       (-2)    /* negative response to FSUC_PROBE */
#define FSUR_IO_SUCCESS         (-3)    /* mount, unmount, repair succeeded */
#define FSUR_IO_FAIL            (-4)    /* unrecoverable I/O error */
#define FSUR_IO_UNCLEAN         (-5)    /* mount failed, file system not clean
	                                 */
#define FSUR_INVAL              (-6)    /* invalid argument */
#define FSUR_LOADERR            (-7)    /* kern_loader error */
#define FSUR_INITRECOGNIZED     (-8)    /* response to FSUC_PROBE or
	                                 * FSUC_PROBEFORINIT, implies that
	                                 * initialization is possible */

/*
 *	mount parameters passed from WSM to the .util program.
 */
#define DEVICE_READONLY         "readonly"
#define DEVICE_WRITABLE         "writable"

#define DEVICE_REMOVABLE        "removable"
#define DEVICE_FIXED            "fixed"

#endif  /* _SYS_LOADABLE_FS_ */
