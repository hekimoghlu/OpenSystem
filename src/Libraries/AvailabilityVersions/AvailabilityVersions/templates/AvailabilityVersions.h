/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
#ifndef __AVAILABILITY_VERSIONS__
#define __AVAILABILITY_VERSIONS__

// @@AVAILABILITY_VERSION_DEFINES()@@


/*
 * Set up standard Mac OS X versions
 */

#if (!defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)) || defined(_DARWIN_C_SOURCE)

// @@ALIAS_VERSION_MACROS(macos, MAC_OS_X_VERSION_,__MAC_,maxVersion=0x000B0000)@@
// @@ALIAS_VERSION_MACROS(macos, MAC_OS_VERSION_,__MAC_,minVersion=0x000B0000)@@

#endif /* #if (!defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)) || defined(_DARWIN_C_SOURCE) */

#endif /* __AVAILABILITY_VERSIONS__ */

