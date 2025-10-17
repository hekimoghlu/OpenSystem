/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#include "kxld_srcversion.h"

/*******************************************************************************
*******************************************************************************/
void
kxld_srcversion_init_from_macho(KXLDsrcversion *srcversion, struct source_version_command *src)
{
	check(srcversion);
	check(src);

	srcversion->version = src->version;
	srcversion->has_srcversion = TRUE;
}

/*******************************************************************************
*******************************************************************************/
void
kxld_srcversion_clear(KXLDsrcversion *srcversion)
{
	bzero(srcversion, sizeof(*srcversion));
}

/*******************************************************************************
*******************************************************************************/
u_long
kxld_srcversion_get_macho_header_size(void)
{
	return sizeof(struct source_version_command);
}

/*******************************************************************************
*******************************************************************************/
kern_return_t
kxld_srcversion_export_macho(const KXLDsrcversion *srcversion, u_char *buf,
    u_long *header_offset, u_long header_size)
{
	kern_return_t rval = KERN_FAILURE;
	struct source_version_command *srcversionhdr = NULL;

	check(srcversion);
	check(buf);
	check(header_offset);

	require_action(sizeof(*srcversionhdr) <= header_size - *header_offset, finish,
	    rval = KERN_FAILURE);
	srcversionhdr = (struct source_version_command *) ((void *) (buf + *header_offset));
	*header_offset += sizeof(*srcversionhdr);

	srcversionhdr->cmd = LC_SOURCE_VERSION;
	srcversionhdr->cmdsize = (uint32_t) sizeof(*srcversionhdr);
	srcversionhdr->version = srcversion->version;

	rval = KERN_SUCCESS;

finish:
	return rval;
}
