/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#ifndef _FSGETPATH_PRIVATE_H_
#define _FSGETPATH_PRIVATE_H_

#ifndef KERNEL

#include <sys/appleapiopts.h>
#include <sys/cdefs.h>
#include <sys/_types/_ssize_t.h>
#include <sys/_types/_size_t.h>
#include <sys/_types/_fsid_t.h>
#include <_types/_uint32_t.h>
#include <_types/_uint64_t.h>
#include <Availability.h>

/*
 * These are only included for compatibility with previous header
 */
#include <sys/types.h>
#include <sys/mount.h>
#ifdef __APPLE_API_PRIVATE
#include <sys/attr.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif  /* __APPLE_API_PRIVATE */

#include <sys/attr_private.h>
#include <sys/_types/_fsobj_id_t.h>

__BEGIN_DECLS

#ifdef __APPLE_API_PRIVATE


/*
 * openbyid_np: open a file given a file system id and a file system object id
 *
 * fsid :	value corresponding to getattlist ATTR_CMN_FSID attribute, or
 *			value of stat's st.st_dev ; set fsid = {st.st_dev, 0}
 *
 * objid: value (link id/node id) corresponding to getattlist ATTR_CMN_OBJID
 *		  attribute , or
 *		  value of stat's st.st_ino (node id); set objid =  st.st_ino
 *
 * For hfs the value of getattlist ATTR_CMN_FSID is a link id which uniquely identifies a
 * parent in the case of hard linked files; this allows unique path access validation.
 * Not all file systems support getattrlist ATTR_CMN_OBJID (link id).
 * A node id does not uniquely identify a parent in the case of hard linked files and may
 * resolve to a path for which access validation can fail.
 */
int openbyid_np(fsid_t* fsid, fsobj_id_t* objid, int flags);

ssize_t fsgetpath_ext(char *, size_t, fsid_t *, uint64_t, uint32_t) __OSX_AVAILABLE(10.15) __IOS_AVAILABLE(13.0) __TVOS_AVAILABLE(13.0) __WATCHOS_AVAILABLE(6.0);

#endif /* __APPLE_API_PRIVATE */

__END_DECLS

#endif /* KERNEL */

#endif /* !_FSGETPATH_PRIVATE_H_ */
