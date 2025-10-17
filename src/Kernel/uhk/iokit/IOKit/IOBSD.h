/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#ifndef _IOBSD_H
#define _IOBSD_H

/*
 * bsd-related registry properties
 */

#define kIOBSDKey      "IOBSD"     // (BSD subsystem resource)
#define kIOBSDNameKey  "BSD Name"  // (an OSString)
#define kIOBSDNamesKey "BSD Names" // (an OSDictionary of OSString's, for links)
#define kIOBSDMajorKey "BSD Major" // (an OSNumber)
#define kIOBSDMinorKey "BSD Minor" // (an OSNumber)
#define kIOBSDUnitKey  "BSD Unit"  // (an OSNumber)


#ifdef KERNEL_PRIVATE

#include <stdint.h>
#include <kern/task.h>

#ifdef __cplusplus
extern "C" {
#endif

struct IOPolledFileIOVars;
struct mount;
struct vnode;

enum{
	kIOMountChangeMount      = 0x00000101,
	kIOMountChangeUnmount    = 0x00000102,
	kIOMountChangeWillResize = 0x00000201,
	kIOMountChangeDidResize  = 0x00000202,
};
extern void IOBSDMountChange(struct mount *mp, uint32_t op);
extern void IOBSDLowSpaceUnlinkKernelCore(void);
/*
 *       Tests that the entitlement is present and true
 */
extern boolean_t IOCurrentTaskHasEntitlement(const char * entitlement);
extern boolean_t IOTaskHasEntitlement(task_t task, const char *entitlement);
extern boolean_t IOVnodeHasEntitlement(struct vnode *vnode, int64_t off, const char *entitlement);
extern boolean_t IOVnodeGetBooleanEntitlement(
	struct vnode *vnode,
	int64_t off,
	const char *entitlement,
	bool *value);
extern char * IOCurrentTaskGetEntitlement(const char * entitlement);
extern char * IOTaskGetEntitlement(task_t task, const char * entitlement);
/*
 * IOVnodeGetEntitlement returns a null-terminated string that must be freed with kfree_data().
 */
extern char *IOVnodeGetEntitlement(struct vnode *vnode, int64_t offset, const char *entitlement);

/*
 *       Tests that the entitlement is present and has matching value
 */
extern boolean_t IOCurrentTaskHasStringEntitlement(const char *entitlement, const char *value);
extern boolean_t IOTaskHasStringEntitlement(task_t task, const char *entitlement, const char *value);

typedef enum {
	kIOPolledCoreFileModeNotInitialized,
	kIOPolledCoreFileModeDisabled,
	kIOPolledCoreFileModeClosed,
	kIOPolledCoreFileModeUnlinked,
	kIOPolledCoreFileModeStackshot,
	kIOPolledCoreFileModeCoredump,
} IOPolledCoreFileMode_t;

extern struct IOPolledFileIOVars * gIOPolledCoreFileVars;
extern kern_return_t gIOPolledCoreFileOpenRet;
extern IOPolledCoreFileMode_t gIOPolledCoreFileMode;

#ifdef __cplusplus
}
#endif

#endif /* XNU_KERNEL_PRIVATE */

#endif /* !_IOBSD_H */
