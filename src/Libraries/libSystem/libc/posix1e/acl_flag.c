/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#include <sys/appleapiopts.h>
#include <sys/types.h>
#include <sys/acl.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "aclvar.h"

#if __DARWIN_ACL_ENTRY_INHERITED != KAUTH_ACE_INHERITED
#  error __DARWIN_ACL_ENTRY_INHERITED != KAUTH_ACE_INHERITED
#endif
#if __DARWIN_ACL_ENTRY_FILE_INHERIT != KAUTH_ACE_FILE_INHERIT
#  error __DARWIN_ACL_ENTRY_FILE_INHERIT != KAUTH_ACE_FILE_INHERIT
#endif
#if __DARWIN_ACL_ENTRY_DIRECTORY_INHERIT != KAUTH_ACE_DIRECTORY_INHERIT
#  error __DARWIN_ACL_ENTRY_DIRECTORY_INHERIT != KAUTH_ACE_DIRECTORY_INHERIT
#endif
#if __DARWIN_ACL_ENTRY_LIMIT_INHERIT != KAUTH_ACE_LIMIT_INHERIT
#  error __DARWIN_ACL_ENTRY_LIMIT_INHERIT != KAUTH_ACE_LIMIT_INHERIT
#endif
#if __DARWIN_ACL_ENTRY_ONLY_INHERIT != KAUTH_ACE_ONLY_INHERIT
#  error __DARWIN_ACL_ENTRY_ONLY_INHERIT != KAUTH_ACE_ONLY_INHERIT
#endif
#if __DARWIN_ACL_FLAG_NO_INHERIT != KAUTH_ACL_NO_INHERIT
#  error __DARWIN_ACL_FLAG_NO_INHERIT != KAUTH_ACL_NO_INHERIT
#endif

int
acl_add_flag_np(acl_flagset_t flags, acl_flag_t flag)
{
	/* XXX validate flags */
	/* XXX validate flag */

	flags->af_flags |= flag;
	return(0);
}

int
acl_clear_flags_np(acl_flagset_t flags)
{
	/* XXX validate flags */

	flags->af_flags = 0;
	return(0);
}

int
acl_delete_flag_np(acl_flagset_t flags, acl_flag_t flag)
{
	/* XXX validate flags */
	/* XXX validate flag */

	flags->af_flags &= ~flag;
	return(0);
}

int
acl_get_flag_np(acl_flagset_t flagset, acl_flag_t flag)
{
	/* XXX validate flags */
	/* XXX validate flag */

	return((flag & flagset->af_flags) ? 1 : 0);
}

int
acl_get_flagset_np(void *obj, acl_flagset_t *flagset_p)
{
	struct _acl		*ap = (struct _acl *)obj;
	struct _acl_entry	*ep = (struct _acl_entry *)obj;
	
	if (_ACL_VALID_ACL(ap)) {
		*flagset_p = (acl_flagset_t)&ap->a_flags;
	} else if (_ACL_VALID_ENTRY(ep)) {
		*flagset_p = (acl_flagset_t)&ep->ae_flags;
	} else {
		errno = EINVAL;
		return(-1);
	}
	return(0);
}

int
acl_set_flagset_np(void *obj, acl_flagset_t flagset)
{
	struct _acl		*ap = (struct _acl *)obj;
	struct _acl_entry	*ep = (struct _acl_entry *)obj;
	
	if (_ACL_VALID_ACL(ap)) {
		ap->a_flags = flagset->af_flags;
	} else if (_ACL_VALID_ENTRY(ep)) {
		ep->ae_flags = flagset->af_flags;
	} else {
		errno = EINVAL;
		return(-1);
	}

	return(0);
}
