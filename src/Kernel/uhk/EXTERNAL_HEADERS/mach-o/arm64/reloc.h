/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
/*
 * Relocation types used in the arm64 implementation.
 */
enum reloc_type_arm64
{
    ARM64_RELOC_UNSIGNED,	  // for pointers
    ARM64_RELOC_SUBTRACTOR,       // must be followed by a ARM64_RELOC_UNSIGNED
    ARM64_RELOC_BRANCH26,         // a B/BL instruction with 26-bit displacement
    ARM64_RELOC_PAGE21,           // pc-rel distance to page of target
    ARM64_RELOC_PAGEOFF12,        // offset within page, scaled by r_length
    ARM64_RELOC_GOT_LOAD_PAGE21,  // pc-rel distance to page of GOT slot
    ARM64_RELOC_GOT_LOAD_PAGEOFF12, // offset within page of GOT slot,
                                    //  scaled by r_length
    ARM64_RELOC_POINTER_TO_GOT,   // for pointers to GOT slots
    ARM64_RELOC_TLVP_LOAD_PAGE21, // pc-rel distance to page of TLVP slot
    ARM64_RELOC_TLVP_LOAD_PAGEOFF12, // offset within page of TLVP slot,
                                     //  scaled by r_length
    ARM64_RELOC_ADDEND		  // must be followed by PAGE21 or PAGEOFF12
};
