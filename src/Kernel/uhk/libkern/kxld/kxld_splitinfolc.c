/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#include "kxld_splitinfolc.h"

/*******************************************************************************
*******************************************************************************/
void
kxld_splitinfolc_init_from_macho(KXLDsplitinfolc *splitinfolc, struct linkedit_data_command *src)
{
	check(splitinfolc);
	check(src);

	splitinfolc->cmdsize = src->cmdsize;
	splitinfolc->dataoff = src->dataoff;
	splitinfolc->datasize = src->datasize;
	splitinfolc->has_splitinfolc = TRUE;
}

/*******************************************************************************
*******************************************************************************/
void
kxld_splitinfolc_clear(KXLDsplitinfolc *splitinfolc)
{
	bzero(splitinfolc, sizeof(*splitinfolc));
}

/*******************************************************************************
*******************************************************************************/
u_long
kxld_splitinfolc_get_macho_header_size(void)
{
	return sizeof(struct linkedit_data_command);
}

/*******************************************************************************
*******************************************************************************/
kern_return_t
kxld_splitinfolc_export_macho(const KXLDsplitinfolc *splitinfolc,
    splitKextLinkInfo *linked_object,
    u_long *header_offset,
    u_long header_size,
    u_long *data_offset,
    u_long size)
{
	kern_return_t       rval = KERN_FAILURE;
	struct linkedit_data_command *splitinfolc_hdr = NULL;
	u_char *            buf;

	check(splitinfolc);
	check(linked_object);
	check(header_offset);
	check(data_offset);

	buf = (u_char *)(linked_object->linkedKext);
	require_action(sizeof(*splitinfolc_hdr) <= header_size - *header_offset,
	    finish,
	    rval = KERN_FAILURE);
	splitinfolc_hdr = (struct linkedit_data_command *)((void *)(buf + *header_offset));
	*header_offset += sizeof(*splitinfolc_hdr);

	if (buf + *data_offset > buf + size) {
		kxld_log(kKxldLogLinking, kKxldLogErr,
		    "\n OVERFLOW! linkedKext %p to %p (%lu) copy %p to %p (%u) <%s>",
		    (void *) buf,
		    (void *) (buf + size),
		    size,
		    (void *) (buf + *data_offset),
		    (void *) (buf + *data_offset + splitinfolc->datasize),
		    splitinfolc->datasize,
		    __func__);
		goto finish;
	}

	// copy in the split info reloc data from kextExecutable. For example dataoff
	// in LC_SEGMENT_SPLIT_INFO load command points to the reloc data in the
	// __LINKEDIT segment.  In this case 65768 into the kextExecutable file is
	// the split seg reloc info (for 920 bytes)
//    Load command 9
//    cmd LC_SEGMENT_SPLIT_INFO
//    cmdsize 16
//    dataoff 65768
//    datasize 920


	memcpy(buf + *data_offset, linked_object->kextExecutable + splitinfolc->dataoff, splitinfolc->datasize);

#if SPLIT_KEXTS_DEBUG
	u_char *dataPtr = buf + *data_offset;

	kxld_log(kKxldLogLinking, kKxldLogErr,
	    "\n\n linkedKext %p to %p (%lu) copy %p to %p (%u) <%s>",
	    (void *) buf,
	    (void *) (buf + size),
	    size,
	    (void *) (dataPtr),
	    (void *) (dataPtr + splitinfolc->datasize),
	    splitinfolc->datasize,
	    __func__);

	if (*(dataPtr + 0) != 0x7F) {
		kxld_log(kKxldLogLinking, kKxldLogErr,
		    "\n\n bad LC_SEGMENT_SPLIT_INFO: 0x%02X %02X %02X %02X %02X %02X %02X %02X at %p (buf %p + %lu) <%s>",
		    *(dataPtr + 0),
		    *(dataPtr + 1),
		    *(dataPtr + 2),
		    *(dataPtr + 3),
		    *(dataPtr + 4),
		    *(dataPtr + 5),
		    *(dataPtr + 6),
		    *(dataPtr + 7),
		    (void *) dataPtr,
		    (void *) buf,
		    *data_offset, __func__);
	}
#endif

	// update the load command header
	splitinfolc_hdr->cmd = LC_SEGMENT_SPLIT_INFO;
	splitinfolc_hdr->cmdsize = (uint32_t) sizeof(*splitinfolc_hdr);
	splitinfolc_hdr->dataoff = (uint32_t)(*data_offset);
	splitinfolc_hdr->datasize = splitinfolc->datasize;

	*data_offset += splitinfolc->datasize;

	rval = KERN_SUCCESS;

finish:
	return rval;
}
