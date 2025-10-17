/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#include <string.h>
#include <mach-o/loader.h>
#include <sys/types.h>

#define DEBUG_ASSERT_COMPONENT_NAME_STRING "kxld"
#include <AssertMacros.h>

#include "kxld_util.h"
#include "kxld_uuid.h"

/*******************************************************************************
*******************************************************************************/
void
kxld_uuid_init_from_macho(KXLDuuid *uuid, struct uuid_command *src)
{
	check(uuid);
	check(src);

	memcpy(uuid->uuid, src->uuid, sizeof(uuid->uuid));
	uuid->has_uuid = TRUE;
}

/*******************************************************************************
*******************************************************************************/
void
kxld_uuid_clear(KXLDuuid *uuid)
{
	bzero(uuid, sizeof(*uuid));
}

/*******************************************************************************
*******************************************************************************/
u_long
kxld_uuid_get_macho_header_size(void)
{
	return sizeof(struct uuid_command);
}

/*******************************************************************************
*******************************************************************************/
kern_return_t
kxld_uuid_export_macho(const KXLDuuid *uuid, u_char *buf,
    u_long *header_offset, u_long header_size)
{
	kern_return_t rval = KERN_FAILURE;
	struct uuid_command *uuidhdr = NULL;

	check(uuid);
	check(buf);
	check(header_offset);

	require_action(sizeof(*uuidhdr) <= header_size - *header_offset, finish,
	    rval = KERN_FAILURE);
	uuidhdr = (struct uuid_command *) ((void *) (buf + *header_offset));
	*header_offset += sizeof(*uuidhdr);

	uuidhdr->cmd = LC_UUID;
	uuidhdr->cmdsize = (uint32_t) sizeof(*uuidhdr);
	memcpy(uuidhdr->uuid, uuid->uuid, sizeof(uuidhdr->uuid));

	rval = KERN_SUCCESS;

finish:
	return rval;
}
