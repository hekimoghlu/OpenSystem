/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#ifndef NVRAM_HELPER_H
#define NVRAM_HELPER_H

#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>

typedef enum{
	OP_SET = 0,
	OP_GET,
	OP_DEL,
	OP_DEL_RET,
	OP_RES,
	OP_OBL,
	OP_SYN
} nvram_op;

#define SystemNVRAMGuidString "40A0DDD2-77F8-4392-B4A3-1E7304206516"
#define CommonNVRAMGuidString "7C436110-AB2A-4BBB-A880-FE41995C9F82"
#define RandomNVRAMGuidString "11112222-77F8-4392-B4A3-1E7304206516"

#define KernelOnlyVariablePrefix "krn."
#define kIONVRAMForceSyncNowPropertyKey "IONVRAM-FORCESYNCNOW-PROPERTY"
#define DefaultSetVal         "1234"

io_registry_entry_t CreateOptionsRef(void);
void ReleaseOptionsRef(io_registry_entry_t optionsRef);
void TestVarOp(nvram_op op, const char *var, const char *val, kern_return_t exp_ret, io_registry_entry_t optionsRef);
CFTypeID GetVarType(const char *name, io_registry_entry_t optionsRef);
#endif /* NVRAM_HELPER_H */
