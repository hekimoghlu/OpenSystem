/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
#ifndef SYS_DOC_TOMBSTONE_H_
#define SYS_DOC_TOMBSTONE_H_

#include <sys/types.h>
#include <stdbool.h>

#ifdef KERNEL_PRIVATE

/*
 * struct representing a document "tombstone" that's recorded
 * when a thread manipulates files marked with a document-id.
 * if the thread recreates the same item, this tombstone is
 * used to preserve the document_id on the new file.
 *
 * It is a separate structure because of its size - we want to
 * allocate it on demand instead of just stuffing it into the
 * uthread structure.
 */
struct doc_tombstone {
	struct vnode     *t_lastop_parent;
	struct vnode     *t_lastop_item;
	uint32_t                  t_lastop_parent_vid;
	uint32_t                  t_lastop_item_vid;
	uint64_t          t_lastop_fileid;
	uint64_t          t_lastop_document_id;
	unsigned char     t_lastop_filename[NAME_MAX + 1];
};

struct componentname;

struct doc_tombstone *doc_tombstone_get(void);
void doc_tombstone_clear(struct doc_tombstone *ut, struct vnode **old_vpp);
void doc_tombstone_save(struct vnode *dvp, struct vnode *vp,
    struct componentname *cnp, uint64_t doc_id,
    ino64_t file_id);
bool doc_tombstone_should_ignore_name(const char *nameptr, int len);
bool doc_tombstone_should_save(struct doc_tombstone *ut, struct vnode *vp,
    struct componentname *cnp);

#endif // defined(KERNEL_PRIVATE)

#endif // SYS_DOC_TOMBSTONE_H_
