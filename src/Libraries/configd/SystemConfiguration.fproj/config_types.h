/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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
#ifndef _CONFIG_TYPES_H
#define _CONFIG_TYPES_H

#include <TargetConditionals.h>

/*
 * Keep IPC functions private to the framework
 */
#ifdef mig_external
#undef mig_external
#endif
#define mig_external __private_extern__

/* Turn MIG type checking on by default */
#ifdef __MigTypeCheck
#undef __MigTypeCheck
#endif
#define __MigTypeCheck	1

/*
 * Mach server port name
 */
#if	!TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
#define SCD_SERVER	"com.apple.SystemConfiguration.configd"
#else	// !TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST
#define SCD_SERVER_HOST	"com.apple.SystemConfiguration.configd"
#define SCD_SERVER	"com.apple.SystemConfiguration.configd_sim"
#endif	// !TARGET_OS_SIMULATOR || TARGET_OS_MACCATALYST

/*
 * Input arguments: serialized key's, list delimiters, ...
 *	(sent as out-of-line data in a message)
 */
typedef const void * xmlData_t;

/* Output arguments: serialized data, lists, ...
 *	(sent as out-of-line data in a message)
 */
typedef const void * xmlDataOut_t;

#endif	/* !_CONFIG_TYPES_H */
