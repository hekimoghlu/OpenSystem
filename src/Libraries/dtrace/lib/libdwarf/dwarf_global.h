/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
typedef struct Dwarf_Global_Context_s *Dwarf_Global_Context;

/* 
    This struct contains header information for a set of pubnames.
    Essentially, they contain the context for a set of pubnames 
    belonging to a compilation-unit.

    This is also used for the sgi-specific
    weaknames, typenames, varnames, funcnames data:
    the structs for those are incomplete and
    instances of this are used instead.

    Also used for DWARF3 .debug_pubtypes.

*/
struct Dwarf_Global_Context_s {

    /* Length in .debug_pubnames (etc) of a set of names for a
       compilation-unit. Dwarf_Word pu_length; The value is not made
       available outside libdwarf and not used inside, so no need to
       record it. */

    /* For this context, size of a length. 4 or 8 */
    unsigned char pu_length_size;

    /* For this CU, size of the extension 0 except for dwarf2 extension 
       64bit, in which case is 4. */
    unsigned char pu_extension_size;

    /* 
       Offset into .debug_info of the compilation-unit header (not DIE) 
       for this set of pubnames. */
    Dwarf_Off pu_offset_of_cu_header;

    /* Size of compilation-unit that these pubnames are in. */
    Dwarf_Unsigned pu_info_length;

    Dwarf_Debug pu_dbg;
};


/* This struct contains information for a single pubname. */
struct Dwarf_Global_s {

    /* 
       Offset from the start of the corresponding compilation-unit of
       the DIE for the given pubname CU. */
    Dwarf_Off gl_named_die_offset_within_cu;

    /* Points to the given pubname. */
    Dwarf_Small *gl_name;

    /* Context for this pubname. */
    Dwarf_Global_Context gl_context;
};

int _dwarf_internal_get_pubnames_like_data(Dwarf_Debug dbg,
					   Dwarf_Small *
					   section_data_ptr,
					   Dwarf_Unsigned
					   section_length,
					   Dwarf_Global ** globals,
					   Dwarf_Signed * return_count,
					   Dwarf_Error * error,
					   int context_code,
					   int global_code,
					   int length_err_num,
					   int version_err_num);

void
_dwarf_internal_globals_dealloc( Dwarf_Debug dbg, Dwarf_Global *dwgl,
        Dwarf_Signed count,
        int context_code,
        int global_code,
        int list_code);


#ifdef __sgi  /* __sgi should only be defined for IRIX/MIPS. */
void _dwarf_fix_up_offset_irix(Dwarf_Debug dbg,
        Dwarf_Unsigned *varp,
        char *caller_site_name);
#define FIX_UP_OFFSET_IRIX_BUG(ldbg,var,name) _dwarf_fix_up_offset_irix(ldbg,&var,name)
#else
#define FIX_UP_OFFSET_IRIX_BUG(ldbg,var,name)
#endif

