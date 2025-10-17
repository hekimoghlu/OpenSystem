/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#ifndef __UAPI_POSIX_ACL_XATTR_H
#define __UAPI_POSIX_ACL_XATTR_H
#include <linux/types.h>
#define POSIX_ACL_XATTR_VERSION 0x0002
#define ACL_UNDEFINED_ID (- 1)
struct posix_acl_xattr_entry {
  __le16 e_tag;
  __le16 e_perm;
  __le32 e_id;
};
struct posix_acl_xattr_header {
  __le32 a_version;
};
#endif
