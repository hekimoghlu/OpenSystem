/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#ifndef fskit_support_h
#define fskit_support_h

typedef enum : int {
    check_fs_op,
    mount_fs_op,
    format_fs_op,
} fskit_command_t;

/*
 * invoke_tool_from_fskit - fsck_, mount_, or newfs_ using FSKit
 *
 *      This routine determines if FSKit is present, and if so,
 * attempts to invoke the tool using the supplied arguments.
 *
 *      This routine returns if FSKit is unavailable, if the named
 * FSModule is unknown, or if the named FSModule does not support this tool.
 *
 *      In case of successful tool invocation or syntax error, this
 * routine exits the calling program.
 *
 *      In the mount_fs_op case, this function will add "nofollow"
 * if MNT_NOFOLLOW is set in flags and the module supports it.
 */
int
invoke_tool_from_fskit(fskit_command_t operation, int flags,
                       int argc, char * const *argv);

#endif /* fskit_support_h */
