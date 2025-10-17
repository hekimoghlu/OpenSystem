/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#ifndef CSHIMS_FILEMANAGER_H
#define CSHIMS_FILEMANAGER_H

#include "_CShimsMacros.h"

#if __has_include(<sys/param.h>)
#include <sys/param.h>
#endif

#if __has_include(<fts.h>)
#include <fts.h>
#endif

#if __has_include(<sys/quota.h>)
#include <sys/quota.h>
#endif

#if __has_include(<sys/xattr.h>)
#include <sys/xattr.h>
#endif

#if __has_include(<dirent.h>)
#include <dirent.h>
#endif

#if __has_include(<removefile.h>)
#include <removefile.h>
#endif // __has_include(<removefile.h>)

#if FOUNDATION_FRAMEWORK && __has_include(<sys/types.h>)
#include <sys/types.h>
// Darwin-specific API that is implemented but not declared in any header
// This function behaves exactly like the public mkpath_np(3) API, but it also returns the first directory it actually created, which helps us make sure we set the given attributes on the right directories.
extern int _mkpath_np(const char *path, mode_t omode, const char **firstdir);
#endif

#endif // CSHIMS_FILEMANAGER_H
