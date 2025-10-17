/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
 * @OSF_COPYRIGHT@
 */

#ifndef _KERN_IPC_MIG_H_
#define _KERN_IPC_MIG_H_

#include <mach/mig.h>
#include <mach/mach_types.h>
#include <mach/message.h>
#include <kern/kern_types.h>

#include <sys/cdefs.h>

#ifdef  XNU_KERNEL_PRIVATE

#include <sys/kdebug.h>

struct ipc_kmsg;

/*
 * Define the trace points for MIG-generated calls.  One traces the input parameters
 * to MIG called things, another traces the outputs, and one traces bad message IDs.
 */
#ifdef _MIG_TRACE_PARAMETERS_

#define __BeforeRcvCallTrace(msgid, arg1, arg2, arg3, arg4)         \
	KERNEL_DEBUG_CONSTANT_IST(KDEBUG_TRACE,                     \
	    KDBG_MIGCODE(msgid) | DBG_FUNC_START,                   \
	    (unsigned int)(arg1),                                   \
	    (unsigned int)(arg2),                                   \
	    (unsigned int)(arg3),                                   \
	    (unsigned int)(arg4),                                   \
	    (unsigned int)(0));

#define __AfterRcvCallTrace(msgid, arg1, arg2, arg3, arg4)          \
	KERNEL_DEBUG_CONSTANT_IST(KDEBUG_TRACE,                     \
	    KDBG_MIGCODE(msgid) | DBG_FUNC_END,                     \
	    (unsigned int)(arg1),                                   \
	    (unsigned int)(arg2),                                   \
	    (unsigned int)(arg3),                                   \
	    (unsigned int)(arg4),                                   \
	    (unsigned int)(0));

#define __BeforeSimpleCallTrace(msgid, arg1, arg2, arg3, arg4)      \
	KERNEL_DEBUG_CONSTANT_IST(KDEBUG_TRACE,                     \
	    KDBG_MIGCODE(msgid) | DBG_FUNC_START,                   \
	    (unsigned int)(arg1),                                   \
	    (unsigned int)(arg2),                                   \
	    (unsigned int)(arg3),                                   \
	    (unsigned int)(arg4),                                   \
	    (unsigned int)(0));

#define __AfterSimpleCallTrace(msgid, arg1, arg2, arg3, arg4)       \
	KERNEL_DEBUG_CONSTANT_IST(KDEBUG_TRACE,                     \
	    KDBG_MIGCODE(msgid) | DBG_FUNC_END,                     \
	    (unsigned int)(arg1),                                   \
	    (unsigned int)(arg2),                                   \
	    (unsigned int)(arg3),                                   \
	    (unsigned int)(arg4),                                   \
	    (unsigned int)(0));

#define __BeforeKobjectServerTrace(msgid)       ((void)0)
#define __AfterKobjectServerTrace(msgid)        ((void)0)

#else /* !_MIG_TRACE_PARAMETERS_ */

#define __BeforeKobjectServerTrace(msgid)                           \
	KERNEL_DEBUG_CONSTANT_IST(KDEBUG_TRACE,                     \
	    KDBG_MIGCODE(msgid) | DBG_FUNC_START, 0u, 0u, 0u, 0u, 0u)

#define __AfterKobjectServerTrace(msgid)                            \
	KERNEL_DEBUG_CONSTANT_IST(KDEBUG_TRACE,                     \
	    KDBG_MIGCODE(msgid) | DBG_FUNC_END, 0u, 0u, 0u, 0u, 0u)

#endif /* !_MIG_TRACE_PARAMETERS_ */

#define _MIG_MSGID_INVALID(msgid)                                   \
	KERNEL_DEBUG_CONSTANT_IST(KDEBUG_TRACE,                     \
	    MACHDBG_CODE(DBG_MACH_MSGID_INVALID, (msgid)), 0u, 0u, 0u, 0u, 0u)

#endif  /* XNU_KERNEL_PRIVATE */

__BEGIN_DECLS

/* Send a message from the kernel */

#if XNU_KERNEL_PRIVATE
extern mach_msg_return_t mach_msg_send_from_kernel(
	mach_msg_header_t       *msg,
	mach_msg_size_t         send_size);
#else
extern mach_msg_return_t mach_msg_send_from_kernel_proper(
	mach_msg_header_t       *msg,
	mach_msg_size_t         send_size);

#define mach_msg_send_from_kernel mach_msg_send_from_kernel_proper
#endif

/*
 * Allocate kernel message buffer(s) large enough to fit the message,
 * and accept a block that populates the message content.
 *
 *     - descriptor_count: Descriptor count in the outgoing message. If non-zero,
 *       kernel expects a complex message, and vice-versa.
 *
 *     - payload_size: Size of data not processed by kernel (i.e. size of data
 *       following the header, or last descriptor if the message is complex).
 *       This is NOT the total size of the outgoing message.
 *
 * For memory safety the kmsg buffers allocated may occupy non-contiguous memory
 * and it is the caller's responsibility to correctly set up the message
 * based on content's offset from the mach message header, as follows:
 *
 * void (^builder)(mach_msg_header_t *header, mach_msg_descriptor_t *descs, void *payload))
 *
 *            header* â”€â”€â”€â”€â”€â”€â”€â–º â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
 *                             â”‚                         â”‚
 *                             â”‚    mach_msg_header_t    â”‚
 *                             â”‚                         â”‚
 *                             â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
 *                             â”‚(optional)mach_msg_body_tâ”‚
 *             descs* â”€â”€â”€â”€â”€â”€â”€â–º â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
 *                             â”‚                         â”‚
 *                             â”‚                         â”‚
 *                             â”‚                         â”‚
 *                             â”‚  (optional)descriptors  â”‚
 *                             â”‚                         â”‚
 *                             â”‚                         â”‚
 *                             â”‚                         â”‚
 *                             â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
 *                              ~~~~~~~~void space~~~~~~~
 *           payload* â”€â”€â”€â”€â”€â”€â”€â–º â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
 *                             â”‚                         â”‚
 *                             â”‚                         â”‚
 *                             â”‚    other data in msg    â”‚
 *                             â”‚                         â”‚
 *                             â”‚                         â”‚
 *                             â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
 *
 *     header: Points to the start of the message, caller should populate this
 *          with the mach message header.
 *
 *     descs: Pointers to the start of kernel descriptors region, if caller requested
 *          a non-zero descriptor_count. Or NULL if descriptor_count is 0 (meaning
 *          the messsage is not complex). Caller should populate this with descriptors.
 *
 *     payload: Points to the start of data not processed by kernel, which is after
 *          the last descriptor, if caller requested a non-zero payload_size. Or
 *          NULL if payload_size is 0.
 *
 *     Mach message body (descriptor count) will be set after the builder accordingly.
 */
#if MACH_KERNEL_PRIVATE || !XNU_KERNEL_PRIVATE

extern mach_msg_return_t kernel_mach_msg_send_with_builder(
	mach_msg_size_t         descriptor_count,
	mach_msg_size_t         payload_size,    /* Warning: NOT total send size */
	void                  (^builder)(mach_msg_header_t *header,
	mach_msg_descriptor_t  *__counted_by(descriptor_count)descs, /* Nullable */
	void                   *__sized_by(payload_size)payload));   /* Nullable */

#endif /* MACH_KERNEL_PRIVATE || !XNU_KERNEL_PRIVATE */
#if XNU_KERNEL_PRIVATE

extern mach_msg_return_t mach_msg_rpc_from_kernel(
	mach_msg_header_t      *msg,
	mach_msg_size_t         send_size,
	mach_msg_size_t         rcv_size);

extern void mach_msg_destroy_from_kernel(
	mach_msg_header_t      *msg);

#else

extern mach_msg_return_t mach_msg_rpc_from_kernel_proper(
	mach_msg_header_t      *msg,
	mach_msg_size_t         send_size,
	mach_msg_size_t         rcv_size);

extern void mach_msg_destroy_from_kernel_proper(
	mach_msg_header_t      *msg);

#define mach_msg_rpc_from_kernel    mach_msg_rpc_from_kernel_proper
#define mach_msg_destroy_from_kernel mach_msg_destroy_from_kernel_proper

#endif

#ifdef XNU_KERNEL_PRIVATE

mach_msg_return_t kernel_mach_msg_rpc(
	mach_msg_header_t      *msg,
	mach_msg_size_t         send_size,
	mach_msg_size_t         rcv_size,
	boolean_t               interruptible,
	boolean_t              *message_moved);

extern mach_msg_return_t kernel_mach_msg_send(
	mach_msg_header_t      *msg,
	mach_msg_size_t         send_size,
	mach_msg_option64_t     option,
	mach_msg_timeout_t      timeout_val,
	boolean_t              *message_moved);

extern mach_msg_return_t kernel_mach_msg_send_kmsg(
	struct ipc_kmsg        *kmsg);

extern mach_msg_return_t kernel_mach_msg_send_with_builder_internal(
	mach_msg_size_t         desc_count,
	mach_msg_size_t         payload_size, /* Not total send size */
	mach_msg_option64_t     option,
	mach_msg_timeout_t      timeout_val,
	boolean_t              *message_moved,
	void                  (^builder)(mach_msg_header_t *,
	mach_msg_descriptor_t *, void *));

extern mach_msg_return_t mach_msg_send_from_kernel_with_options(
	mach_msg_header_t      *msg,
	mach_msg_size_t         send_size,
	mach_msg_option64_t     option,
	mach_msg_timeout_t      timeout_val);

#endif /* XNU_KERNEL_PRIVATE */
#ifdef  MACH_KERNEL_PRIVATE

extern void mach_msg_receive_continue(void);

#endif  /* MACH_KERNEL_PRIVATE */

__END_DECLS

#endif  /* _KERN_IPC_MIG_H_ */
