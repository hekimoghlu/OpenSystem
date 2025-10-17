/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
/* !!ONLY FOR USE INTERNALLY TO LIBARCHIVE!! */

#ifndef ARCHIVE_PLATFORM_ACL_H_INCLUDED
#define ARCHIVE_PLATFORM_ACL_H_INCLUDED

#ifndef __LIBARCHIVE_BUILD
#ifndef __LIBARCHIVE_TEST_COMMON
#error This header is only to be used internally to libarchive.
#endif
#endif

/*
 * Determine what ACL types are supported
 */
#if ARCHIVE_ACL_FREEBSD || ARCHIVE_ACL_SUNOS || ARCHIVE_ACL_LIBACL
#define ARCHIVE_ACL_POSIX1E     1
#endif

#if ARCHIVE_ACL_FREEBSD_NFS4 || ARCHIVE_ACL_SUNOS_NFS4 || \
    ARCHIVE_ACL_DARWIN  || ARCHIVE_ACL_LIBRICHACL
#define ARCHIVE_ACL_NFS4        1
#endif

#if ARCHIVE_ACL_POSIX1E || ARCHIVE_ACL_NFS4
#define ARCHIVE_ACL_SUPPORT     1
#endif

#endif	/* ARCHIVE_PLATFORM_ACL_H_INCLUDED */
