/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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
    This struct holds information about a abbreviation.
    It is put in the hash table for abbreviations for
    a compile-unit.
*/
struct Dwarf_Abbrev_List_s {

    Dwarf_Word ab_code;
    Dwarf_Half ab_tag;
    Dwarf_Half ab_has_child;

    /* 
       Points to start of attribute and form pairs in the .debug_abbrev 
       section for the abbrev. */
    Dwarf_Byte_Ptr ab_abbrev_ptr;

    struct Dwarf_Abbrev_List_s *ab_next;
};
