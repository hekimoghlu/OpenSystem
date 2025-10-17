/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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
#include "kxld_versionmin.h"

/*******************************************************************************
*******************************************************************************/
void
kxld_versionmin_init_from_macho(KXLDversionmin *versionmin, struct version_min_command *src)
{
	check(versionmin);
	check(src);
	check((src->cmd == LC_VERSION_MIN_MACOSX) || (src->cmd == LC_VERSION_MIN_IPHONEOS) || (src->cmd == LC_VERSION_MIN_TVOS) || (src->cmd == LC_VERSION_MIN_WATCHOS));

	switch (src->cmd) {
	case LC_VERSION_MIN_MACOSX:
		versionmin->platform = kKxldVersionMinMacOSX;
		break;
	case LC_VERSION_MIN_IPHONEOS:
		versionmin->platform = kKxldVersionMiniPhoneOS;
		break;
	case LC_VERSION_MIN_TVOS:
		versionmin->platform = kKxldVersionMinAppleTVOS;
		break;
	case LC_VERSION_MIN_WATCHOS:
		versionmin->platform = kKxldVersionMinWatchOS;
		break;
	}

	versionmin->version = src->version;
	versionmin->has_versionmin = TRUE;
}

void
kxld_versionmin_init_from_build_cmd(KXLDversionmin *versionmin, struct build_version_command *src)
{
	check(versionmin);
	check(src);
	switch (src->platform) {
	case PLATFORM_MACOS:
		versionmin->platform = kKxldVersionMinMacOSX;
		break;
	case PLATFORM_IOS:
		versionmin->platform = kKxldVersionMiniPhoneOS;
		break;
	case PLATFORM_TVOS:
		versionmin->platform = kKxldVersionMinAppleTVOS;
		break;
	case PLATFORM_WATCHOS:
		versionmin->platform = kKxldVersionMinWatchOS;
		break;
	default:
		return;
	}
	versionmin->version = src->minos;
	versionmin->has_versionmin = TRUE;
}

/*******************************************************************************
*******************************************************************************/
void
kxld_versionmin_clear(KXLDversionmin *versionmin)
{
	bzero(versionmin, sizeof(*versionmin));
}

/*******************************************************************************
*******************************************************************************/
u_long
kxld_versionmin_get_macho_header_size(__unused const KXLDversionmin *versionmin)
{
	/* TODO: eventually we can just use struct build_version_command */
	return sizeof(struct version_min_command);
}

/*******************************************************************************
*******************************************************************************/
kern_return_t
kxld_versionmin_export_macho(const KXLDversionmin *versionmin, u_char *buf,
    u_long *header_offset, u_long header_size)
{
	kern_return_t rval = KERN_FAILURE;
	struct version_min_command *versionminhdr = NULL;

	check(versionmin);
	check(buf);
	check(header_offset);


	require_action(sizeof(*versionminhdr) <= header_size - *header_offset, finish,
	    rval = KERN_FAILURE);
	versionminhdr = (struct version_min_command *) ((void *) (buf + *header_offset));
	bzero(versionminhdr, sizeof(*versionminhdr));
	*header_offset += sizeof(*versionminhdr);

	switch (versionmin->platform) {
	case kKxldVersionMinMacOSX:
		versionminhdr->cmd = LC_VERSION_MIN_MACOSX;
		break;
	case kKxldVersionMiniPhoneOS:
		versionminhdr->cmd = LC_VERSION_MIN_IPHONEOS;
		break;
	case kKxldVersionMinAppleTVOS:
		versionminhdr->cmd = LC_VERSION_MIN_TVOS;
		break;
	case kKxldVersionMinWatchOS:
		versionminhdr->cmd = LC_VERSION_MIN_WATCHOS;
		break;
	default:
		goto finish;
	}
	versionminhdr->cmdsize = (uint32_t) sizeof(*versionminhdr);
	versionminhdr->version = versionmin->version;
	versionminhdr->sdk = 0;

	rval = KERN_SUCCESS;

finish:
	return rval;
}
