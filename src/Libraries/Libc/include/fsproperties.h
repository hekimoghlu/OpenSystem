/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#ifndef _FSPROPERTIES_H_
#define _FSPROPERTIES_H_

/* Info plist keys */
#define kFSMediaTypesKey             "FSMediaTypes"
#define kFSPersonalitiesKey          "FSPersonalities"

/* Sub-keys for FSMediaTypes dictionaries */
#define kFSMediaPropertiesKey        "FSMediaProperties"
#define kFSProbeArgumentsKey         "FSProbeArguments"
#define kFSProbeExecutableKey        "FSProbeExecutable"
#define kFSProbeOrderKey             "FSProbeOrder"

/* Sub-keys for FSPersonalities dictionaries */
#define kFSFormatArgumentsKey        "FSFormatArguments"
#define kFSFormatContentMaskKey      "FSFormatContentMask"
#define kFSFormatExecutableKey       "FSFormatExecutable"
#define kFSFormatInteractiveKey      "FSFormatInteractive"
#define kFSFormatMinimumSizeKey      "FSFormatMinimumSize"
#define kFSFormatMaximumSizeKey      "FSFormatMaximumSize"
#define kFSMountArgumentsKey         "FSMountArguments"
#define kFSMountExecutableKey        "FSMountExecutable"
#define kFSNameKey                   "FSName"
#define kFSRepairArgumentsKey        "FSRepairArguments"
#define kFSRepairExecutableKey       "FSRepairExecutable"
#define kFSVerificationArgumentsKey  "FSVerificationArguments"
#define kFSVerificationExecutableKey "FSVerificationExecutable"
#define kFSSubTypeKey                "FSSubType"
#define kFSXMLOutputArgumentKey      "FSXMLOutputArgument"

#define kFSEncryptNameKey 			 "FSEncryptionName"
	/* Deprecated - use kFSEncryptNameKey for HFS/APFS */
#define	kFSCoreStorageEncryptNameKey "FSCoreStorageEncryptionName"

#endif /* _FSPROPERTIES_H_ */
