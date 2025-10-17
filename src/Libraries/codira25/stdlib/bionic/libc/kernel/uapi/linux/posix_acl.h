/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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
#ifndef __UAPI_POSIX_ACL_H
#define __UAPI_POSIX_ACL_H
#define ACL_UNDEFINED_ID (- 1)
#define ACL_TYPE_ACCESS (0x8000)
#define ACL_TYPE_DEFAULT (0x4000)
#define ACL_USER_OBJ (0x01)
#define ACL_USER (0x02)
#define ACL_GROUP_OBJ (0x04)
#define ACL_GROUP (0x08)
#define ACL_MASK (0x10)
#define ACL_OTHER (0x20)
#define ACL_READ (0x04)
#define ACL_WRITE (0x02)
#define ACL_EXECUTE (0x01)
#endif
