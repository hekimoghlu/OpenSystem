/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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
/* DO NOT MODIFY THIS FILE
 *
 * This file is a partial mirror of
 *    AppleIntelCPUPowerManagement/pmioctl.h
 * Changes may only be made to the original, pmioctl.h.
 * This file must be updated only when pmioctl.h changes.
 */

/*
 * Defines the IOCTLs for dealing with the CPU power management KEXT.
 */
#ifndef _IOPMROOTDOMAINIOCTLS_H_
#define _IOPMROOTDOMAINIOCTLS_H_

#include <sys/ioccom.h>
#include <i386/pmCPU.h>

#define PMIOCGETVARIDINFO       _IOW('P', 25, uint64_t)
#define PMIOCGETVARNAMEINFO     _IOW('P', 26, uint64_t)
#define PMIOCSETVARINFO         _IOW('P', 27, uint64_t)

/*
 * Data structures used by IOCTLs
 */
#pragma pack(4)

#define PMVARNAMELEN    16

typedef enum{
	vUnknown            = 0,        /* Unknown type */
	vBool               = 1,        /* Boolean value */
	vInt                = 2,        /* signed integer value */
	vUInt               = 3,        /* Unsigned integer value */
	vChars              = 4,        /* 8 characters */
	vInvalid            = -1        /* invalid type */
} pmioctlVarType_t;

typedef struct pmioctlVaribleInfo {
	uint32_t            varID;      /* ID of variable */
	uint8_t             varName[PMVARNAMELEN + 1];
	pmioctlVarType_t    varType;    /* type of variable's value */
	uint64_t            varInitValue;/* variable's initial value */
	uint64_t            varCurValue;/* variable's current value */
} pmioctlVariableInfo_t;

#pragma pack()

#endif /* _IOPMROOTDOMAINIOCTLS_H_ */
