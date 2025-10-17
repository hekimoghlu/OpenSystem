/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
/* This structure is used to read an arange into. */
struct Dwarf_Arange_s {

    /* Starting address of the arange, ie low-pc. */
    Dwarf_Addr ar_address;

    /* Length of the arange. */
    Dwarf_Unsigned ar_length;

    /* 
       Offset into .debug_info of the start of the compilation-unit
       containing this set of aranges. */
    Dwarf_Off ar_info_offset;

    /* Corresponding Dwarf_Debug. */
    Dwarf_Debug ar_dbg;
};



int
  _dwarf_get_aranges_addr_offsets(Dwarf_Debug dbg,
				  Dwarf_Addr ** addrs,
				  Dwarf_Off ** offsets,
				  Dwarf_Signed * count,
				  Dwarf_Error * error);
