/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#ifndef _FSGETPATH_H_
#define _FSGETPATH_H_

#ifndef KERNEL

#include <machine/_types.h>
#include <sys/_types/_ssize_t.h>
#include <sys/_types/_size_t.h>
#include <sys/_types/_fsid_t.h>
#include <_types/_uint64_t.h>
#include <Availability.h>

__BEGIN_DECLS

/*
 * Obtain the full pathname of a file system object by id.
 */
ssize_t fsgetpath(char *, size_t, fsid_t *, uint64_t) __OSX_AVAILABLE(10.13) __IOS_AVAILABLE(11.0) __TVOS_AVAILABLE(11.0) __WATCHOS_AVAILABLE(4.0);

__END_DECLS

#endif /* KERNEL */

#if PRIVATE
#include <sys/fsgetpath_private.h>
#endif

#endif /* !_FSGETPATH_H_ */
