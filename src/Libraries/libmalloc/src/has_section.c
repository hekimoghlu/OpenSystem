/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#include "internal.h"

#if MALLOC_TARGET_64BIT

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <string.h>

// Copied from libmacho/getsecbyname.c, because we do not want to add a
// dependency on cctools/libmacho (-lmacho).  Only sync with original, never
// modify independently.
// Version, git hash/tag: cctools-987
static
// *** DO NOT MODIFY - START ***
/*
 * This routine returns the section structure for the named section in the
 * named segment for the mach_header_64 pointer passed to it if it exist.
 * Otherwise it returns zero.
 */
const struct section_64 *
my_getsectbynamefromheader_64(
struct mach_header_64 *mhp,
const char *segname,
const char *sectname)
{
	struct segment_command_64 *sgp;
	struct section_64 *sp;
	uint32_t i, j;
        
	sgp = (struct segment_command_64 *)
	      ((char *)mhp + sizeof(struct mach_header_64));
	for(i = 0; i < mhp->ncmds; i++){
	    if(sgp->cmd == LC_SEGMENT_64)
		if(strncmp(sgp->segname, segname, sizeof(sgp->segname)) == 0 ||
		   mhp->filetype == MH_OBJECT){
		    sp = (struct section_64 *)((char *)sgp +
			 sizeof(struct segment_command_64));
		    for(j = 0; j < sgp->nsects; j++){
			if(strncmp(sp->sectname, sectname,
			   sizeof(sp->sectname)) == 0 &&
			   strncmp(sp->segname, segname,
			   sizeof(sp->segname)) == 0)
			    return(sp);
			sp = (struct section_64 *)((char *)sp +
			     sizeof(struct section_64));
		    }
		}
	    sgp = (struct segment_command_64 *)((char *)sgp + sgp->cmdsize);
	}
	return((struct section_64 *)0);
}
// *** DO NOT MODIFY - END ***
#endif  // MALLOC_TARGET_64BIT

MALLOC_NOEXPORT
bool
main_image_has_section(const char* segname, const char *sectname)
{
#if MALLOC_TARGET_64BIT
	const struct mach_header* mh = _dyld_get_image_header(0);
	return my_getsectbynamefromheader_64((struct mach_header_64 *)mh, segname, sectname) != NULL;
#else
	return false;
#endif
}
