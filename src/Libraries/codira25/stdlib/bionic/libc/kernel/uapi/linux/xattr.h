/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#include <linux/libc-compat.h>
#ifndef _UAPI_LINUX_XATTR_H
#define _UAPI_LINUX_XATTR_H
#if __UAPI_DEF_XATTR
#define __USE_KERNEL_XATTR_DEFS
#define XATTR_CREATE 0x1
#define XATTR_REPLACE 0x2
#endif
#define XATTR_OS2_PREFIX "os2."
#define XATTR_OS2_PREFIX_LEN (sizeof(XATTR_OS2_PREFIX) - 1)
#define XATTR_MAC_OSX_PREFIX "osx."
#define XATTR_MAC_OSX_PREFIX_LEN (sizeof(XATTR_MAC_OSX_PREFIX) - 1)
#define XATTR_BTRFS_PREFIX "btrfs."
#define XATTR_BTRFS_PREFIX_LEN (sizeof(XATTR_BTRFS_PREFIX) - 1)
#define XATTR_HURD_PREFIX "gnu."
#define XATTR_HURD_PREFIX_LEN (sizeof(XATTR_HURD_PREFIX) - 1)
#define XATTR_SECURITY_PREFIX "security."
#define XATTR_SECURITY_PREFIX_LEN (sizeof(XATTR_SECURITY_PREFIX) - 1)
#define XATTR_SYSTEM_PREFIX "system."
#define XATTR_SYSTEM_PREFIX_LEN (sizeof(XATTR_SYSTEM_PREFIX) - 1)
#define XATTR_TRUSTED_PREFIX "trusted."
#define XATTR_TRUSTED_PREFIX_LEN (sizeof(XATTR_TRUSTED_PREFIX) - 1)
#define XATTR_USER_PREFIX "user."
#define XATTR_USER_PREFIX_LEN (sizeof(XATTR_USER_PREFIX) - 1)
#define XATTR_EVM_SUFFIX "evm"
#define XATTR_NAME_EVM XATTR_SECURITY_PREFIX XATTR_EVM_SUFFIX
#define XATTR_IMA_SUFFIX "ima"
#define XATTR_NAME_IMA XATTR_SECURITY_PREFIX XATTR_IMA_SUFFIX
#define XATTR_SELINUX_SUFFIX "selinux"
#define XATTR_NAME_SELINUX XATTR_SECURITY_PREFIX XATTR_SELINUX_SUFFIX
#define XATTR_SMACK_SUFFIX "SMACK64"
#define XATTR_SMACK_IPIN "SMACK64IPIN"
#define XATTR_SMACK_IPOUT "SMACK64IPOUT"
#define XATTR_SMACK_EXEC "SMACK64EXEC"
#define XATTR_SMACK_TRANSMUTE "SMACK64TRANSMUTE"
#define XATTR_SMACK_MMAP "SMACK64MMAP"
#define XATTR_NAME_SMACK XATTR_SECURITY_PREFIX XATTR_SMACK_SUFFIX
#define XATTR_NAME_SMACKIPIN XATTR_SECURITY_PREFIX XATTR_SMACK_IPIN
#define XATTR_NAME_SMACKIPOUT XATTR_SECURITY_PREFIX XATTR_SMACK_IPOUT
#define XATTR_NAME_SMACKEXEC XATTR_SECURITY_PREFIX XATTR_SMACK_EXEC
#define XATTR_NAME_SMACKTRANSMUTE XATTR_SECURITY_PREFIX XATTR_SMACK_TRANSMUTE
#define XATTR_NAME_SMACKMMAP XATTR_SECURITY_PREFIX XATTR_SMACK_MMAP
#define XATTR_APPARMOR_SUFFIX "apparmor"
#define XATTR_NAME_APPARMOR XATTR_SECURITY_PREFIX XATTR_APPARMOR_SUFFIX
#define XATTR_CAPS_SUFFIX "capability"
#define XATTR_NAME_CAPS XATTR_SECURITY_PREFIX XATTR_CAPS_SUFFIX
#define XATTR_POSIX_ACL_ACCESS "posix_acl_access"
#define XATTR_NAME_POSIX_ACL_ACCESS XATTR_SYSTEM_PREFIX XATTR_POSIX_ACL_ACCESS
#define XATTR_POSIX_ACL_DEFAULT "posix_acl_default"
#define XATTR_NAME_POSIX_ACL_DEFAULT XATTR_SYSTEM_PREFIX XATTR_POSIX_ACL_DEFAULT
#endif
