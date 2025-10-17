/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
/* fsck_msgnums.h
 *
 * This file contain fsck status message numbers associated with 
 * each fsck message string.  These status message numbers and their 
 * strings are file system independent.  
 */

#ifndef __FSCK_MSGNUMS_H
#define __FSCK_MSGNUMS_H

/* Generic fsck status message numbers.  These are file system 
 * independent messages that indicate the current state of verify or 
 * repair run or provide information about damaged files/folder.   
 *
 * The corresponding strings and the mapping array of message number 
 * and other attributes exists in fsck_strings.c
 */
enum fsck_msgnum {
    fsckUnknown                         = 100,

    fsckCheckingVolume                  = 101,  /* Checking volume */
    fsckRecheckingVolume                = 102,  /* Rechecking volume */
    fsckRepairingVolume                 = 103,  /* Repairing volume */
    fsckVolumeOK                        = 104,  /* The volume %s appears to be OK */
    fsckRepairSuccessful                = 105,  /* The volume %s was repaired successfully */
    fsckVolumeVerifyIncomplete          = 106,  /* The volume %s could not be verified completely */
    fsckVolumeVerifyIncompleteNoRepair  = 107,  /* The volume %s could not be verified completely and can not be repaired */
    fsckVolumeCorruptNoRepair           = 108,  /* The volume %s was found corrupt and can not be repaired */
    fsckVolumeCorruptNeedsRepair        = 109,  /* The volume %s was found corrupt and needs to be repaired */
    fsckVolumeNotRepaired               = 110,  /* The volume %s could not be repaired */

    fsckVolumeNotRepairedInUse          = 111,  /* The volume %s cannot be repaired when it is in use */
    fsckVolumeNotVerifiedInUse          = 112,  /* The volume %s cannot be verified when it is in use */
    fsckFileFolderDamage                = 113,  /* File/folder %s may be damaged */
    fsckFileFolderNotRepaired           = 114,  /* File/folder %s could not be repaired */
    fsckVolumeNotRepairedTries          = 115,  /* The volume %s could not be repaired after %d attempts */
    fsckLostFoundDirectory              = 116,  /* Look for missing items in %s directory */
    fsckCorruptFilesDirectory           = 117,  /* Look for links to corrupt files in %s directory */
    fsckInformation                     = 118,  /* Using %s (version %s) for checking volume %s of type %s. */
    fsckProgress                        = 119,  /* %d */
    fsckTrimming                        = 120,  /* Trimming unused blocks */
    fsckVolumeName                      = 121,	/* The volume name is %s */
    fsckVolumeModified			= 122,	/* The volume was modified */
    fsckLimitedRepairs			= 123,	/* Limited repair mode, not all repairs available */
};

#endif
