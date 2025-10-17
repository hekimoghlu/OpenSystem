/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
 * attr.h - attribute data structures and interfaces
 *
 * Copyright (c) 1998, Apple Computer, Inc.  All Rights Reserved.
 */

#ifndef _SYS_ATTR_PRIVATE_H_
#define _SYS_ATTR_PRIVATE_H_

#include <sys/appleapiopts.h>
#include <sys/attr.h>

#ifdef __APPLE_API_UNSTABLE

#define FSOPT_EXCHANGE_DATA_ONLY 0x0000010

#define FSOPT_LIST_SNAPSHOT     0x00000040
#endif /* __APPLE_API_UNSTABLE */
#define FSOPT_NOFIRMLINKPATH     0x00000080
#ifdef __APPLE_API_UNSTABLE
#define FSOPT_FOLLOW_FIRMLINK    0x00000100
#endif /* __APPLE_API_UNSTABLE */
#define FSOPT_ISREALFSID         FSOPT_RETURN_REALDEV
#ifdef __APPLE_API_UNSTABLE
#define FSOPT_UTIMES_NULL        0x00000400

/* Volume supports kqueue notifications for remote events */
#define VOL_CAP_INT_REMOTE_EVENT                0x00008000

#endif /* __APPLE_API_UNSTABLE */
#endif /* !_SYS_ATTR_PRIVATE_H_ */
