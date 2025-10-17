/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#ifndef _KERN_SMR_TYPES_H_
#define _KERN_SMR_TYPES_H_

#include <sys/cdefs.h>
#include <stdbool.h>
#include <stdint.h>
#include <os/base.h>

__BEGIN_DECLS

/*!
 * @typedef smr_seq_t
 *
 * @brief
 * Represents an opaque SMR sequence number.
 */
typedef unsigned long           smr_seq_t;

/*!
 * @typedef smr_t
 *
 * @brief
 * Type for an SMR domain.
 */
typedef struct smr             *smr_t;


/*!
 * @typedef smr_node_t
 *
 * @brief
 * Intrusive data structure used with @c ssmr_call() to defer callbacks
 * to a safe time.
 */
typedef struct smr_node        *smr_node_t;

/*!
 * @typedef smr_cb_t
 *
 * @brief
 * A callback acting on an @c smr_node_t to destroy it.
 */
typedef void (*smr_cb_t)(smr_node_t);

struct smr_node {
	struct smr_node        *smrn_next;
	smr_cb_t XNU_PTRAUTH_SIGNED_FUNCTION_PTR("ssmr_cb_t") smrn_cb;
};

/*!
 * @macro SMR_POINTER_DECL
 *
 * @brief
 * Macro to declare a pointer type that uses SMR for access.
 */
#define SMR_POINTER_DECL(name, type_t) \
	struct name { type_t volatile __smr_ptr; }

/*!
 * @macro SMR_POINTER
 *
 * @brief
 * Macro to declare a pointer that uses SMR for access.
 */
#define SMR_POINTER(type_t) \
	SMR_POINTER_DECL(, type_t)


/* internal types that clients should not use directly */
typedef SMR_POINTER(struct smrq_slink *) __smrq_slink_t;
typedef SMR_POINTER(struct smrq_link *)  __smrq_link_t;


/*!
 * @struct smrq_slink
 *
 * @brief
 * Type used to represent a linkage in an SMR queue
 * (single form, with O(n) deletion).
 */
struct smrq_slink {
	__smrq_slink_t          next;
};

/*!
 * @struct smrq_link
 *
 * @brief
 * Type used to represent a linkage in an SMR queue
 * (double form, with O(1) deletion).
 */
struct smrq_link {
	__smrq_link_t           next;
	__smrq_link_t          *prev;
};


/*!
 * @struct smrq_slist_head
 *
 * @brief
 * Type used to represent the head of a singly linked list.
 *
 * @discussion
 * This must be used with @c smrq_slink linkages.
 *
 * This type supports:
 * - insertion at the head,
 * - O(n) removal / replacement.
 */
struct smrq_slist_head {
	__smrq_slink_t          first;
};

#define SMRQ_SLIST_INITIALIZER(name) \
	{ .first = { NULL } }

/*!
 * @struct smrq_list_head
 *
 * @brief
 * Type used to represent the head of a doubly linked list.
 *
 * @discussion
 * This must be used with @c smrq_link linkages.
 *
 * This type supports:
 * - insertion at the head,
 * - O(1) removal / replacement.
 */
struct smrq_list_head {
	__smrq_link_t           first;
};

#define SMRQ_LIST_INITIALIZER(name) \
	{ .first = { NULL } }

/*!
 * @struct smrq_stailq_head
 *
 * @brief
 * Type used to represent the head of a singly linked tail-queue.
 *
 * @discussion
 * This must be used with @c smrq_slink linkages.
 *
 * This type supports:
 * - insertion at the head,
 * - insertion at the tail,
 * - O(n) removal / replacement.
 */
struct smrq_stailq_head {
	__smrq_slink_t          first;
	__smrq_slink_t         *last;
};

#define SMRQ_STAILQ_INITIALIZER(name) \
	{ .first = { NULL }, .last = &(name).first }

/*!
 * @struct smrq_tailq_head
 *
 * @brief
 * Type used to represent the head of a doubly linked tail-queue.
 *
 * @discussion
 * This must be used with @c smrq_link linkages.
 *
 * This type supports:
 * - insertion at the head,
 * - insertion at the tail,
 * - O(1) removal / replacement.
 */
struct smrq_tailq_head {
	__smrq_link_t           first;
	__smrq_link_t          *last;
};

#define SMRQ_TAILQ_INITIALIZER(name) \
	{ .first = { NULL }, .last = &(name).first }

__END_DECLS

#endif /* _KERN_SMR_TYPES_H_ */
