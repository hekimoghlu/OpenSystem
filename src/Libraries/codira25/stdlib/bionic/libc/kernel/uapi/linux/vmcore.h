/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#ifndef _UAPI_VMCORE_H
#define _UAPI_VMCORE_H
#include <linux/types.h>
#define VMCOREDD_NOTE_NAME "LINUX"
#define VMCOREDD_MAX_NAME_BYTES 44
struct vmcoredd_header {
  __u32 n_namesz;
  __u32 n_descsz;
  __u32 n_type;
  __u8 name[8];
  __u8 dump_name[VMCOREDD_MAX_NAME_BYTES];
};
#endif
