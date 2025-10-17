/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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

#if __DARWIN_ACL_EXTENDED_ALLOW != KAUTH_ACE_PERMIT
#  error __DARWIN_ACL_EXTENDED_ALLOW != KAUTH_ACE_PERMIT
#endif
#if __DARWIN_ACL_EXTENDED_DENY != KAUTH_ACE_DENY
#  error __DARWIN_ACL_EXTENDED_DENY != KAUTH_ACE_DENY
#endif

int
acl_copy_entry(acl_entry_t dest, acl_entry_t src)
{
	/* validate arguments */
	_ACL_VALIDATE_ENTRY(dest);
	_ACL_VALIDATE_ENTRY(src);
	if (dest == src) {
		errno = EINVAL;
		return(-1);
	}
	bcopy(src, dest, sizeof(*src));
	return(0);
}

int
acl_create_entry_np(acl_t *acl_p, acl_entry_t *entry_p, int index)
{
	struct _acl	*ap = *acl_p;
	int		i;

	/* validate arguments */
	_ACL_VALIDATE_ACL(ap);
	if (ap->a_entries >= ACL_MAX_ENTRIES) {
		errno = ENOMEM;
		return(-1);
	}
	if (index == ACL_LAST_ENTRY)
		index = ap->a_entries;
	if (index > ap->a_entries) {
		errno = ERANGE;
		return(-1);
	}

	/* move following entries out of the way */
	for (i = ap->a_entries; i > index; i--)
		ap->a_ace[i] = ap->a_ace[i - 1];
	ap->a_entries++;

	/* initialise new entry */
	memset(&ap->a_ace[index], 0, sizeof(ap->a_ace[index]));
	ap->a_ace[index].ae_magic = _ACL_ENTRY_MAGIC;
	ap->a_ace[index].ae_tag = ACL_UNDEFINED_TAG;

	*entry_p = &ap->a_ace[index];
	return(0);
}

int
acl_create_entry(acl_t *acl_p, acl_entry_t *entry_p)
{
	return(acl_create_entry_np(acl_p, entry_p, ACL_LAST_ENTRY));
}

int
acl_delete_entry(acl_t acl, acl_entry_t entry)
{
	int	i;

	_ACL_VALIDATE_ACL(acl);
	_ACL_VALIDATE_ENTRY(entry);
	_ACL_VALIDATE_ENTRY_CONTAINED(acl, entry);

	/* copy following entries down & invalidate last slot */
	acl->a_entries--;
	for (i = entry - &acl->a_ace[0]; i < acl->a_entries; i++)
		acl->a_ace[i] = acl->a_ace[i + 1];
	acl->a_ace[acl->a_entries].ae_magic = 0;
	/* Sync up the iterator's position if necessary */
	if (acl->a_last_get >= (entry - &acl->a_ace[0]))
	  acl->a_last_get--;

	return(0);
}

int
acl_get_entry(acl_t acl, int entry_id, acl_entry_t *entry_p)
{

	_ACL_VALIDATE_ACL(acl);
	if ((entry_id != ACL_FIRST_ENTRY) &&
	    (entry_id != ACL_NEXT_ENTRY) &&
	    (entry_id != ACL_LAST_ENTRY) &&
	    ((entry_id < 0) || (entry_id >= acl->a_entries))) {
		errno = EINVAL;
		return(-1);
	}
	if (entry_id == ACL_FIRST_ENTRY)
	  entry_id = 0;
	else
	  if (entry_id == ACL_NEXT_ENTRY) {
	    entry_id = acl->a_last_get + 1;
	  }
	  else
	    if (entry_id == ACL_LAST_ENTRY)
	      entry_id = acl->a_entries - 1;

	if (entry_id >= acl->a_entries) {
	  errno = EINVAL;
	  return (-1);
	}

	*entry_p = &acl->a_ace[entry_id];
	acl->a_last_get = entry_id;

	return(0);
}

void *
acl_get_qualifier(acl_entry_t entry)
{
	acl_tag_t	tag_type;
	void		*result;
	int		error;

	result = NULL;
	if (!_ACL_VALID_ENTRY(entry)) {
		errno = EINVAL;
	} else if ((error = acl_get_tag_type(entry, &tag_type)) != 0) {
		/* errno is set by acl_get_tag_type */
	} else {
		switch(tag_type) {
		case ACL_EXTENDED_ALLOW:
		case ACL_EXTENDED_DENY:
			if ((result = malloc(sizeof(guid_t))) != NULL)
				bcopy(&entry->ae_applicable, result, sizeof(guid_t));
			break;
		default:
			errno = EINVAL;
			break;
		}
	}
	return(result);
}

int
acl_get_tag_type(acl_entry_t entry, acl_tag_t *tag_type_p)
{
	_ACL_VALIDATE_ENTRY(entry);

	*tag_type_p = entry->ae_tag;
	return(0);
}

int
acl_set_qualifier(acl_entry_t entry, const void *tag_qualifier_p)
{
	acl_tag_t	tag_type;

	_ACL_VALIDATE_ENTRY(entry);
	if (acl_get_tag_type(entry, &tag_type) != 0)
		return(-1);

	switch(tag_type) {
	case ACL_EXTENDED_ALLOW:
	case ACL_EXTENDED_DENY:
		bcopy(tag_qualifier_p, &entry->ae_applicable, sizeof(guid_t));
		break;
	default:
		errno = EINVAL;
		return(-1);
	}
	return(0);
}

int
acl_set_tag_type(acl_entry_t entry, acl_tag_t tag_type)
{
	_ACL_VALIDATE_ENTRY(entry);

	switch(tag_type) {
	case ACL_EXTENDED_ALLOW:
	case ACL_EXTENDED_DENY:
		entry->ae_tag = tag_type;
		break;
	default:
		errno = EINVAL;
		return(-1);
	}
	return(0);
}
