/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
 * FILE: kextmanager_types.h
 * AUTH: I/O Kit Team (Copyright Apple Computer, 2002, 2006-7)
 * DATE: June 2002, September 2006, August 2007
 * DESC: typedefs for the kextmanager_mig.defs's MiG-generated code 
 *
 */

#ifndef __KEXT_TYPES_H__
#define __KEXT_TYPES_H__

#include <mach/mach_types.h>        // allows to compile standalone
#include <mach/kmod.h>
#include <sys/param.h>
#include <sys/mount.h>
#include <uuid/uuid.h>              // uuid_t

#define KEXTD_SERVER_NAME       "com.apple.KernelExtensionServer"
#define PROPERTYKEY_LEN         128

typedef int kext_result_t;
typedef char mountpoint_t[MNAMELEN];
typedef char property_key_t[PROPERTYKEY_LEN];
typedef char kext_bundle_id_t[KMOD_MAX_NAME];
typedef char posix_path_t[MAXPATHLEN];

// nowadays this is binary plist data but we keep the name for now
typedef char * xmlDataOut_t;
typedef char * xmlDataIn_t;

#endif /* __KEXT_TYPES_H__ */
