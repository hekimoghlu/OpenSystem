/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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
#ifndef _SMBFS_ATTRLIST_H_
#define _SMBFS_ATTRLIST_H_

#include <sys/appleapiopts.h>

#ifdef KERNEL
#ifdef __APPLE_API_PRIVATE
#include <sys/attr.h>
#include <sys/vnode.h>

#include <smbfs/smbfs_lockf.h>
#include <smbfs/smbfs_node.h>


struct attrblock {
	struct attrlist *ab_attrlist;
	void **ab_attrbufpp;
	void **ab_varbufpp;
	int	ab_flags;
	int	ab_blocksize;
	vfs_context_t ab_context;
};

/* 
 * The following define the attributes that HFS supports:
 */

#define SMBFS_ATTR_CMN_VALID				\
	(ATTR_CMN_NAME | ATTR_CMN_DEVID	|		\
	 ATTR_CMN_FSID | ATTR_CMN_OBJTYPE |		\
	 ATTR_CMN_OBJTAG | ATTR_CMN_OBJID |		\
	 ATTR_CMN_OBJPERMANENTID | ATTR_CMN_PAROBJID |	\
	 ATTR_CMN_SCRIPT | ATTR_CMN_CRTIME |		\
	 ATTR_CMN_MODTIME | ATTR_CMN_CHGTIME |		\
	 ATTR_CMN_ACCTIME | ATTR_CMN_BKUPTIME |		\
	 ATTR_CMN_FNDRINFO |ATTR_CMN_OWNERID |		\
	 ATTR_CMN_GRPID | ATTR_CMN_ACCESSMASK |		\
	 ATTR_CMN_FLAGS | ATTR_CMN_USERACCESS |		\
	 ATTR_CMN_FILEID | ATTR_CMN_PARENTID )

#define SMBFS_ATTR_CMN_SEARCH_VALID	\
	(ATTR_CMN_NAME | ATTR_CMN_OBJID |	\
	 ATTR_CMN_PAROBJID | ATTR_CMN_CRTIME |	\
	 ATTR_CMN_MODTIME | ATTR_CMN_CHGTIME | 	\
	 ATTR_CMN_ACCTIME | ATTR_CMN_BKUPTIME |	\
	 ATTR_CMN_FNDRINFO | ATTR_CMN_OWNERID |	\
	 ATTR_CMN_GRPID	| ATTR_CMN_ACCESSMASK | \
	 ATTR_CMN_FILEID | ATTR_CMN_PARENTID ) 



#define SMBFS_ATTR_DIR_VALID				\
	(ATTR_DIR_LINKCOUNT | ATTR_DIR_ENTRYCOUNT | ATTR_DIR_MOUNTSTATUS)

#define SMBFS_ATTR_DIR_SEARCH_VALID	\
	(ATTR_DIR_ENTRYCOUNT)

#define SMBFS_ATTR_FILE_VALID				  \
	(ATTR_FILE_LINKCOUNT |ATTR_FILE_TOTALSIZE |	  \
	 ATTR_FILE_ALLOCSIZE | ATTR_FILE_IOBLOCKSIZE |	  \
	 ATTR_FILE_CLUMPSIZE | ATTR_FILE_DEVTYPE |	  \
	 ATTR_FILE_DATALENGTH | ATTR_FILE_DATAALLOCSIZE | \
	 ATTR_FILE_RSRCLENGTH | ATTR_FILE_RSRCALLOCSIZE)

#define SMBFS_ATTR_FILE_SEARCH_VALID		\
	(ATTR_FILE_DATALENGTH | ATTR_FILE_DATAALLOCSIZE |	\
	 ATTR_FILE_RSRCLENGTH | ATTR_FILE_RSRCALLOCSIZE )

int smbfs_vnop_readdirattr(struct vnop_readdirattr_args *ap);

#endif /* __APPLE_API_PRIVATE */
#endif /* KERNEL */
#endif /* ! _SMBFS_ATTRLIST_H_ */
