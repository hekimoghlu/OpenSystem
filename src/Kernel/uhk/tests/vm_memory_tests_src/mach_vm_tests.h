/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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

#include <unistd.h>
#include <spawn.h>
#include <os/log.h>
#include <bootstrap_priv.h>
#include <libproc.h>
#include <signal.h>
#include <stdatomic.h>
#include <os/assumes.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>
#include <darwintest.h>
#include <darwintest_utils.h>

#include <mach/mach_time.h>
#include <mach/message.h>
#include <mach/mach_traps.h>
#include <mach/mach_vm.h>

#define VM_OP_NONE                    (0)
#define VM_OP_UNMAP                   (1)
#define VM_OP_EXIT                    (2)
#define VM_OP_COPY                    (3)
#define VM_OP_READ                    (4)
#define VM_OP_MEMENTRY                (5)
#define VM_OP_REMAP                   (6)
#define VM_OP_READ_OVERWRITE          (7)
#define VM_OP_WRITE                   (8)
#define VM_OP_EXIT_ERROR              (9)

extern mach_port_t serverPort;
extern mach_port_t persistentReplyPort;
extern boolean_t debug;

struct ipc_message {
	mach_msg_header_t header;
	mach_msg_body_t body;
	mach_msg_port_descriptor_t port_descriptor;
	boolean_t                       copy;
	int                                     vm_op;
	uint64_t                        address;
	uint64_t                        pid;
	uint64_t                        size;
	uint64_t                        misoffset;
	char                            value[64];
};
typedef struct ipc_message ipc_message_t;

void mach_vm_client(mach_port_t);
void mach_server_remap(mach_port_t);
void mach_server_read(mach_port_t, int);
void mach_server_make_memory_entry(mach_port_t);

int mach_server_data_setup(void **);
void mach_server_data_cleanup(void *, mach_vm_address_t, mach_vm_size_t);
void mach_server_construct_header(ipc_message_t *, mach_port_t);
void mach_server_create_allocation(mach_vm_address_t *, mach_vm_size_t, void *);
void mach_server_contruct_payload(ipc_message_t *, mach_vm_address_t, mach_port_t, mach_vm_size_t, mach_vm_offset_t, boolean_t, int);
void server_error_out(mach_port_t);


#define MACH_VM_TEST_SERVICE_NAME  "com.apple.test.xnu.vm.machVMTest"
#define VM_SPAWN_TOOL "/AppleInternal/Tests/xnu/darwintests/tools/vm_spawn_tool"
