/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#ifndef _SYS_CONTENT_PROTECTION_H_
#define _SYS_CONTENT_PROTECTION_H_

#ifdef PRIVATE

/*
 * Protection classes vary in their restrictions on read/writability.  A is generally
 * the strictest, and D is effectively no restriction.
 */

/*
 * dir_none forces new items created in the directory to pick up the mount point default
 * protection level. it is only allowed for directories.
 */
#define PROTECTION_CLASS_DIR_NONE 0

#define PROTECTION_CLASS_A  1
#define PROTECTION_CLASS_B  2
#define PROTECTION_CLASS_C  3
#define PROTECTION_CLASS_D  4
#define PROTECTION_CLASS_E  5
#define PROTECTION_CLASS_F  6
#define PROTECTION_CLASS_CX 7

#define PROTECTION_CLASS_MIN  PROTECTION_CLASS_A
#define PROTECTION_CLASS_MAX  PROTECTION_CLASS_CX

/*
 * This forces open_dprotected_np to behave as though the file were created with
 * the traditional open(2) semantics.
 */
#define PROTECTION_CLASS_DEFAULT  (-1)

#endif /* PRIVATE */

#endif /* _SYS_CONTENT_PROTECTION_H_ */
