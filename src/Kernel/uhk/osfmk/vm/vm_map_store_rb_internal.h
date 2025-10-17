/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#ifndef _VM_VM_MAP_STORE_H_RB
#define _VM_VM_MAP_STORE_H_RB

RB_PROTOTYPE_SC(__private_extern__, rb_head, vm_map_store, entry, rb_node_compare);

extern void vm_map_store_init_rb(
	struct vm_map_header   *header);

extern int rb_node_compare(
	struct vm_map_store    *first,
	struct vm_map_store    *second);

extern bool vm_map_store_lookup_entry_rb(
	struct _vm_map         *map,
	vm_map_offset_t         address,
	struct vm_map_entry   **entryp);

extern void vm_map_store_entry_link_rb(
	struct vm_map_header   *header,
	struct vm_map_entry    *entry);

extern void vm_map_store_entry_unlink_rb(
	struct vm_map_header   *header,
	struct vm_map_entry    *entry);

extern void vm_map_store_copy_reset_rb(
	struct vm_map_copy     *copy_map,
	struct vm_map_entry    *entry,
	int                     nentries);

extern void update_first_free_rb(
	struct _vm_map         *map,
	struct vm_map_entry    *entry,
	bool                    new_entry_creation);

#endif /* _VM_VM_MAP_STORE_RB_H */
