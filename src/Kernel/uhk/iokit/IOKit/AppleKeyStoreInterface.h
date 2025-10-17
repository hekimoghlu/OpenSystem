/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
#ifndef _IOKIT_APPLEKEYSTOREINTERFACE_H
#define _IOKIT_APPLEKEYSTOREINTERFACE_H

// These are currently duplicate defs with different names
// from AppleKeyStore & CoreStorage

// aka MAX_KEY_SIZE
#define AKS_MAX_KEY_SIZE    128

// aka rawKey
struct aks_raw_key_t {
	uint32_t  keybytecount;
	uint8_t   keybytes[AKS_MAX_KEY_SIZE];
};

// aka volumeKey
struct aks_volume_key_t {
	uint32_t      algorithm;
	aks_raw_key_t key;
};

// aka AKS_GETKEY
#define AKS_PLATFORM_FUNCTION_GETKEY    "getKey"

// aka kCSFDETargetVEKID
#define PLATFORM_FUNCTION_GET_MEDIA_ENCRYPTION_KEY_UUID  "CSFDETargetVEKID"

#define AKS_SERVICE_PATH                "/IOResources/AppleFDEKeyStore"

#endif /* _IOKIT_APPLEKEYSTOREINTERFACE_H */
