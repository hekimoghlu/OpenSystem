/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#ifndef	_OCSPD_TYPES_H_
#define _OCSPD_TYPES_H_

#include <mach/mach_types.h>
#include <MacTypes.h>

/* Explicitly enable MIG type checking per Radar 4735696 */
#undef __MigTypeCheck
#define __MigTypeCheck 1

typedef void *Data;

/*
 * Standard bootstrap name and an env var name to override it with (!NDEBUG only)
 */
#define OCSPD_BOOTSTRAP_NAME		"com.apple.ocspd"
#define OCSPD_BOOTSTRAP_ENV			"OCSPD_SERVER"

#endif	/* _OCSPD_TYPES_H_ */
