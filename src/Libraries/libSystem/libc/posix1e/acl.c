/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#include "aclvar.h"

acl_t
acl_dup(acl_t acl)
{
	struct _acl	*ap;

	if (!_ACL_VALID_ACL(acl)) {
		errno = EINVAL;
		return(NULL);
	}
	
	if ((ap = malloc(sizeof(*ap))) != NULL)
		bcopy(acl, ap, sizeof(*ap));
	return(ap);
}

int
acl_free(void *obj)
{
	/*
	 * Without tracking the addresses of text buffers and qualifiers,
	 * we can't validate the obj argument here at all.
	 */
	if(obj != _FILESEC_REMOVE_ACL)
		free(obj);
	return(0);
}

acl_t
acl_init(int count)
{
	struct _acl	*ap;

	/* validate count */
	if (count < 0) {
		errno = EINVAL;
		return(NULL);
	}
	if (count > ACL_MAX_ENTRIES) {
		errno = ENOMEM;
		return(NULL);
	}

	if ((ap = malloc(sizeof (*ap))) != NULL) {
		bzero(ap, sizeof(*ap));
		ap->a_magic = _ACL_ACL_MAGIC;
		ap->a_last_get = -1;
	}
	return(ap);
}

int
acl_valid(acl_t acl)
{
	_ACL_VALIDATE_ACL(acl);

	/* XXX */
	return(0);
}

int
acl_valid_fd_np(int fd, acl_type_t type, acl_t acl)
{
	errno = ENOTSUP;	/* XXX */
	return(-1);
}

int
acl_valid_file_np(const char *path, acl_type_t type, acl_t acl)
{
	errno = ENOTSUP;	/* XXX */
	return(-1);
}

int
acl_valid_link(const char *path, acl_type_t type, acl_t acl)
{
	errno = ENOTSUP;	/* XXX */
	return(-1);
}

/*
 * Not applicable; not supportedl
 */
int
acl_calc_mask(__unused acl_t *acl_p)
{
	errno = ENOTSUP;
	return(-1);
}
